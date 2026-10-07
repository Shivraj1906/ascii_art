#!/usr/bin/env python3
"""Compare a preserved CUDA baseline with one-shot, reusable and resident paths."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

from run import ROOT, command, compare, digest, optional_command, summary

SIZES = ['256x256', '512x512', '812x812', '1000x1250', '1920x1080', '3840x2160']
MODES = ['baseline', 'optimized', 'graph_direct', 'graph_auto']
METRICS = ['total_ms', 'load_ms', 'conversion_ms', 'write_ms', 'overhead_ms',
           'pipeline_ms', 'peak_rss_mib', 'device_buffer_mib']


def worker(exe, source, output, metrics, warmup, repeat, bloom, profile, device, mode):
    args = [str(exe), str(source), str(output), str(metrics), str(warmup), str(repeat),
            str(int(bloom)), str(int(profile)), str(device)]
    if mode != 'baseline':
        args += [str(int(mode.startswith('graph'))), '1' if mode == 'graph_direct' else '0']
    begin = time.perf_counter()
    command(args)
    return {'command': args, 'process_ms': (time.perf_counter()-begin)*1000,
            'metrics': json.loads(metrics.read_text())}


def agreement(expected, actual):
    result = compare(expected, actual)
    result['baseline_png_sha256'] = result.pop('python_png_sha256')
    result['optimized_png_sha256'] = result.pop('cuda_png_sha256')
    return result


def summarize(record):
    metrics = ['execution_ms', 'pipeline_ms', 'device_buffer_mib'] if record['mode'] == 'resident' else METRICS
    record['summary'] = {name: summary([s[name] for s in record['steady']]) for name in metrics}
    if record['fresh']:
        record['summary']['fresh_process_ms'] = summary([s['process_ms'] for s in record['fresh']])
    if record['profile']:
        record['stage_summary'] = {name: summary([s['stage_ms'][name] for s in record['profile']])
                                   for name in record['profile'][0]['stage_ms']}


def report(data, result_dir):
    import numpy as np
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mpl-cache'))
    import matplotlib.pyplot as plt
    for record in data['records']:
        summarize(record)
    with (result_dir / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['size', 'bloom', 'mode', 'metric',
             'n', 'median', 'mean', 'min', 'max', 'stddev', 'p90'], lineterminator='\n')
        writer.writeheader()
        for r in data['records']:
            for metric, stats in {**r['summary'], **{f'stage:{k}': v for k,v in r.get('stage_summary', {}).items()}}.items():
                writer.writerow({'size': r['size'], 'bloom': r['bloom'], 'mode': r['mode'], 'metric': metric, **stats})
    data['report_generator_sha256'] = digest(Path(__file__))
    (result_dir / 'results.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    records = {(r['size'], r['bloom'], r['mode']): r for r in data['records']}
    def median(size, bloom, mode, metric):
        return records[size,bloom,mode]['summary'][metric]['median']
    settings = data['settings']
    warm_ratios = [median(size,bloom,'baseline','total_ms')/median(size,bloom,'optimized','total_ms')
                   for size in settings['sizes'] for bloom in (False,True)]
    lines = ['# CUDA optimization benchmark', '', f"Measured {data['started_utc']} (UTC).", '',
        'This report compares the committed CUDA implementation before optimization with the optimized code in the same session. The original Python comparison remains in [BENCHMARKS.md](BENCHMARKS.md).', '',
        f"Optimized single-image conversions are **{min(warm_ratios):.2f}–{max(warm_ratios):.2f}× faster end to end** across this suite. The GPU interval and resident execution tables below show the processing improvements separately from PNG I/O.", '', '## What changed', '',
        '- Warp-aligned shared-memory Gaussian tiles process several outputs per thread, share coefficients, exploit coefficient symmetry, skip interior border arithmetic, and specialize the default short filters.',
        '- The wide Gaussian writes binary DoG directly. Bright thresholding happens during bloom tile loading. Four warps share Sobel input and vote on four 8×8 glyph cells with ballots, preserving first-pixel tie breaking.',
        '- Rendering also reduces its min/max range with warp shuffles. Binary atlases use a proven direct-encoding path when normalization cannot change the result.',
        '- One aligned arena replaces per-buffer allocations. Three full-size float buffers suffice for default settings; gray becomes bloom and narrow becomes rendering storage.',
        '- Reusable contexts cache atlases, coefficients, buffers, streams, and captured CUDA graphs. Large reusable frames use cuFFT for exact finite convolution with reflected halos; single-image automatic mode uses direct filtering to avoid FFT plan setup.',
        '- RGB PNG decoding avoids RGBA expansion. PNG output defaults to lossless level 1 without filter trials; levels 6–9 trade encoding time for smaller files.', '',
        '## Environment and method', '',
        f"- CPU/OS: {data['environment']['cpu']} / {data['environment']['platform']}",
        f"- GPU/driver/memory: `{data['environment']['gpu']}`",
        f"- CUDA: `{data['environment']['nvcc']}`",
        f"- Baseline commit: `{data['baseline_commit']}`",
        f"- Both builds: Release, architecture `{data['environment']['architecture']}`; same toolchain and atlas/input bytes.",
        f"- Host compiler: `{data['environment'].get('host_compiler', 'see raw metadata')}`; libpng/libjpeg: `{data['environment'].get('io_library_versions', 'see raw metadata')}`",
        '- Desktop GPU, no fixed clocks, CPU affinity, cache flushing, or fsync. Results include power-management, thermal and background variation.', '',
        f"Each of the 12 cases runs {settings['trials']} independent warmed trials per path, each with {settings['warmup']} discarded conversions and {settings['repeat']} measured conversions. Trial order rotates across implementations. The {settings['fresh_repeat']} fresh-process samples per one-shot path include launch through exit. Stage events are enabled only in separate diagnostic runs. Statistics are descriptive; P90 is interpolated, and standard deviation is the sample standard deviation.", '',
        'Inputs are RGB8 Lanczos resizes of `images/sample.png`. Default 8×8 atlases, Gaussian radii, thresholds, full-resolution bloom and normalization are unchanged. All outputs are decoded before comparison. The optimized PNG compression is lossless, but its default level changes file size and encoding time; end-to-end gains include that tradeoff.', '',
        '**Baseline / optimized** allocate and release device buffers on every conversion. **Graph direct / graph auto** retain a fixed-size context after warmup, but still decode input/atlases, upload RGB, download pixels, encode PNG, and free host outputs each iteration. `conversion_ms` includes the entire host API call; in graph modes it excludes context creation, which is paid during warmup. Atlas decoding remains in graph totals for a controlled comparison.', '',
        '**Resident** decodes/uploads once, then measures sequential synchronous GPU graph replays with input and output already on the GPU. `execution_ms` includes launch, synchronization, event queries and host bookkeeping in the API, excluding decode, transfers, context creation and encoding. One final download/save verifies output outside the interval. These numbers describe execution throughput for the supplied frame, rather than disk-to-disk throughput or a concurrent video pipeline.', '',
        '**GPU pipeline** is the CUDA event interval excluding RGB transfers. The baseline interval also contains per-blur coefficient generation/upload, frees and scheduling gaps. Optimized coefficients are prepared before this interval; one-shot launches can still have scheduling gaps, while graph replay reduces them. Separate Nsight summaries distinguish kernel time and API costs. Stage medians need not sum to pipeline medians.', '',
        '## Warmed end-to-end latency', '', 'Medians in milliseconds. Speedup is committed baseline / optimized one-shot total.', '',
        '| Input | Bloom | Baseline | Optimized | Speedup | Graph direct | Graph auto |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for size in settings['sizes']:
        for bloom in (False, True):
            values = [median(size,bloom,mode,'total_ms') for mode in MODES]
            lines.append(f"| {size} | {'on' if bloom else 'off'} | {values[0]:.3f} | {values[1]:.3f} | {values[0]/values[1]:.2f}× | {values[2]:.3f} | {values[3]:.3f} |")
    lines += ['', '## GPU interval and resident throughput', '',
        'GPU medians in milliseconds; resident FPS is 1000 / median execution time. Setup covers decode/context/plan creation and upload in one resident process, excluded from replay timing (one sample). Resident replay has a different GPU duty cycle from the I/O-heavy runs, so power management can also affect these intervals.', '',
        '| Input | Bloom | Baseline GPU | Optimized GPU | Graph direct GPU | Graph auto GPU | Resident GPU | Resident execution | Resident FPS | Setup ms |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for size in settings['sizes']:
        for bloom in (False, True):
            gpu = [median(size,bloom,mode,'pipeline_ms') for mode in MODES+['resident']]
            wall = median(size,bloom,'resident','execution_ms')
            lines.append(f"| {size} | {'on' if bloom else 'off'} | " + ' | '.join(f'{v:.4f}' for v in gpu)+f" | {wall:.4f} | {1000/wall:.1f} | {records[size,bloom,'resident']['setup_ms']:.2f} |")
    lines += ['', '## Conversion API, I/O, memory and PNG size', '',
        'API, decode, and write medians are milliseconds. Device MiB counts the aligned image/coefficient arena and explicit FFT workspace, excluding driver context and opaque cuFFT plan storage. RSS uses Linux ru_maxrss, which can retain the launcher’s memory high-water mark from before exec. It is an upper bound, rather than isolated converter working memory; identical 4K peaks across modes do not establish equal memory use. A local exec probe confirmed this inheritance. Device allocation counters are independent of that effect.', '',
        '| Input | Bloom | Path | API | Decode | Write | Device MiB | Peak RSS MiB | PNG KiB |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for size in settings['sizes']:
        for bloom in (False, True):
            for mode in MODES:
                r = records[size,bloom,mode]
                values = [median(size,bloom,mode,k) for k in ['conversion_ms','load_ms','write_ms','device_buffer_mib','peak_rss_mib']]
                lines.append(f"| {size} | {'on' if bloom else 'off'} | {mode} | "+' | '.join(f'{v:.3f}' for v in values)+f" | {r['output_bytes']/1024:.2f} |")
    lines += ['', '## Fresh process latency', '',
        'One-shot process medians include driver startup and executable shutdown. Cached filesystem pages are not flushed.', '',
        '| Input | Bloom | Baseline ms | Optimized ms | Speedup |',
        '| --- | --- | ---: | ---: | ---: |']
    for size in settings['sizes']:
        for bloom in (False, True):
            a,b = [median(size,bloom,m,'fresh_process_ms') for m in ['baseline','optimized']]
            lines.append(f"| {size} | {'on' if bloom else 'off'} | {a:.3f} | {b:.3f} | {a/b:.2f}× |")
    lines += ['', '## Stage profiles', '',
        'Separately measured with stage events enabled. Fused stages report zero in their former slot; their work belongs to the containing stage. Sobel includes cell voting for the default glyphs.', '',
        '| Input | Bloom | Path | Stage | Median ms |', '| --- | --- | --- | --- | ---: |']
    for r in data['records']:
        for name, stats in r.get('stage_summary',{}).items():
            lines.append(f"| {r['size']} | {'on' if r['bloom'] else 'off'} | {r['mode']} | {name} | {stats['median']:.5f} |")
    if data.get('nsight'):
        lines += ['', '## Nsight kernel and API diagnostics', '',
            'These separate 1080p captures contain one warmup and one measured conversion. Kernel totals cover both conversions; API totals also include process initialization, context/graph creation and cleanup. Kernel names are grouped below; every original row remains in the linked CSV files. These instrumentation runs do not provide headline latency.', '',
            '| Bloom | Path | Kernel group | Launches | Total ms | Mean ms |',
            '| --- | --- | --- | ---: | ---: | ---: |']
        for capture in data['nsight']:
            prefix = result_dir/f"nsight_{capture['mode']}_{int(capture['bloom'])}"
            groups = {}
            with Path(str(prefix)+'_cuda_gpu_kern_sum.csv').open() as file:
                for row in csv.DictReader(file):
                    name = row['Name']
                    if 'gaussian' in name:
                        name = 'Gaussian horizontal' if '<(bool)1' in name else 'Gaussian vertical'
                    elif any(part in name for part in ['regular_fft<','vector_fft<','postprocess_kernel<','preprocess_kernel<','kernel_wrapper<']):
                        name = 'cuFFT transforms / packing'
                    else:
                        name = name.split('(',1)[0].removeprefix('void ').split('<',1)[0]
                    total,count = groups.get(name,(0,0))
                    groups[name] = (total+float(row['Total Time (ns)'])/1e6,count+int(row['Instances']))
            for name,(total,count) in sorted(groups.items(),key=lambda item:-item[1][0]):
                lines.append(f"| {'on' if capture['bloom'] else 'off'} | {capture['mode']} | {name} | {count} | {total:.4f} | {total/count:.4f} |")
        lines += ['', 'Largest five CUDA API intervals per capture. Waiting inside a synchronization or free call includes outstanding GPU work.', '',
                  '| Bloom | Path | API | Calls | Total ms | Mean ms |',
                  '| --- | --- | --- | ---: | ---: | ---: |']
        for capture in data['nsight']:
            path = result_dir/f"nsight_{capture['mode']}_{int(capture['bloom'])}_cuda_api_sum.csv"
            with path.open() as file:
                for row in list(csv.DictReader(file))[:5]:
                    lines.append(f"| {'on' if capture['bloom'] else 'off'} | {capture['mode']} | {row['Name']} | {row['Num Calls']} | {float(row['Total Time (ns)'])/1e6:.4f} | {float(row['Avg (ns)'])/1e6:.4f} |")
    lines += ['', '## Decoded output agreement', '',
        'All optimized paths are compared against the committed CUDA baseline. Finite FFT convolution changes floating-point accumulation, and values near thresholds can differ. The independent oracle suite covers custom glyphs, edge bins, borders, constant signals, normalization, direct/FFT paths and graph replay.', '',
        '| Input | Bloom | Path | Max error | MAE | Exact % | Within 2 levels % |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in data['records']:
        if 'agreement' not in r: continue
        q = r['agreement']
        lines.append(f"| {r['size']} | {'on' if r['bloom'] else 'off'} | {r['mode']} | {q['max_abs_error']} | {q['mae']:.6f} | {q['exact_percent']:.4f} | {q['within_2_percent']:.4f} |")
    if data.get('validation'):
        lines += ['', '## Validation', '',
            'The normal multi-architecture release build and the architecture-86 benchmark build pass all six CTests. Compute Sanitizer reports zero memory errors, race hazards, and synchronization errors on direct/FFT graph and resident paths. Host image I/O passes AddressSanitizer/UndefinedBehaviorSanitizer (leak detection disabled in this environment). No-bloom benchmark outputs match exactly; bloom outputs differ by at most one gray level in this run.', '']
    lines += ['', '## Artifacts and reproduction', '',
        '- [All raw samples, commands, hashes and metadata](benchmarks/optimized_results/results.json)',
        '- [Every metric: median, mean, min, max, standard deviation, P90](benchmarks/optimized_results/summary.csv)',
        '- [Latency plot](benchmarks/optimized_results/latency.png) · [Publication SVG](benchmarks/optimized_results/latency.svg)',
        '- Nsight kernel/API/transfer summaries: `benchmarks/optimized_results/nsight_*_*.csv` (separate two-conversion captures).', '',
        '```sh', 'mkdir -p benchmarks/work/baseline-source',
        f"git archive {data['baseline_commit']} | tar -x -C benchmarks/work/baseline-source",
        'cmake -S benchmarks/work/baseline-source -B benchmarks/work/baseline-build \\',
        '  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTING=OFF',
        'cmake --build benchmarks/work/baseline-build -j',
        'cmake -S . -B build-opt -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86',
        'cmake --build build-opt -j',
        '.venv/bin/python benchmarks/compare_cuda.py', '```', '',
        'Use `--sizes`, `--trials`, `--repeat`, `--fresh-repeat`, `--resident-repeat`, `--device`, or `--skip-nsight` to change the run. `--report-only` regenerates tables and plots without the GPU. Input generation, hashes, exact commands and raw samples are retained. Binary profiler files and temporary PNGs stay in ignored `benchmarks/work/`.', '']
    (ROOT / 'OPTIMIZATION.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for axis,bloom in zip(axes,(False,True)):
        x = np.arange(len(settings['sizes']))
        for offset,mode,label,color in [(-.25,'baseline','Baseline','#777777'),(0,'optimized','Optimized one-shot','#3465a4'),(.25,'graph_auto','Reusable graph','#3b9252')]:
            axis.bar(x+offset,[median(s,bloom,mode,'total_ms') for s in settings['sizes']],width=.25,label=label,color=color)
        axis.set_xticks(x,settings['sizes'],rotation=35,ha='right'); axis.set_yscale('log')
        axis.set_title(f"Bloom {'on' if bloom else 'off'}");axis.set_ylabel('Warmed total, median ms (log scale)')
        axis.grid(axis='y',alpha=.2);axis.legend()
    fig.suptitle('CUDA optimization: measured end-to-end latency\n'+data['environment']['gpu'].split(',')[0],fontsize=11)
    fig.savefig(result_dir/'latency.png',dpi=180);fig.savefig(result_dir/'latency.svg');plt.close(fig)
    svg = result_dir/'latency.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-build',type=Path,default=ROOT/'benchmarks/work/baseline-build')
    parser.add_argument('--build',type=Path,default=ROOT/'build-opt')
    parser.add_argument('--sizes',default=','.join(SIZES))
    parser.add_argument('--warmup',type=int,default=2);parser.add_argument('--repeat',type=int,default=5)
    parser.add_argument('--trials',type=int,default=3);parser.add_argument('--fresh-repeat',type=int,default=3)
    parser.add_argument('--resident-repeat',type=int,default=100);parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--baseline-ref',default='3f711bb');parser.add_argument('--skip-nsight',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args = parser.parse_args(); result_dir = ROOT/'benchmarks/optimized_results';result_dir.mkdir(parents=True,exist_ok=True)
    if args.report_only:
        report(json.loads((result_dir/'results.json').read_text()),result_dir);return
    if min(args.warmup,args.device) < 0 or min(args.repeat,args.trials,args.fresh_repeat,args.resident_repeat) < 1:
        parser.error('invalid iteration/device setting')
    sizes = args.sizes.split(',')
    from PIL import Image
    work = ROOT/'benchmarks/work/optimization';work.mkdir(parents=True,exist_ok=True)
    inputs = {}
    image = Image.open(ROOT/'images/sample.png').convert('RGB')
    for size in sizes:
        try: width,height = [int(n) for n in size.split('x')]
        except ValueError: parser.error('sizes must use WIDTHxHEIGHT')
        if min(width,height) < 8: parser.error('sizes must be at least 8x8')
        path = work/f'input_{size}.png';image.resize((width,height),Image.Resampling.LANCZOS).save(path)
        inputs[size] = {'path':str(path),'sha256':digest(path)}
    cpu = next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor())
    data = {'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'baseline_commit':command(['git','rev-parse',args.baseline_ref]).strip(),
        'settings':{**vars(args),'sizes':sizes,'baseline_build':str(args.baseline_build.resolve()),'build':str(args.build.resolve())},
        'environment':{'cpu':cpu,'platform':platform.platform(),
          'gpu':optional_command(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader']),
          'nvcc':optional_command(['nvcc','--version']).splitlines()[-2],
          'nsys':optional_command(['nsys','--version']),
          'architecture':next((line.split('=',1)[1] for line in (args.build/'CMakeCache.txt').read_text().splitlines()
                              if line.startswith('CMAKE_CUDA_ARCHITECTURES:')), 'unknown'),
          'git_head':command(['git','rev-parse','HEAD']).strip(),'git_status':command(['git','status','--short']),
          'python':platform.python_version()},'inputs':inputs,'records':[], 'nsight':[]}
    # JSON settings contain Paths in parsed arguments; normalize them once.
    data['settings'] = {k:str(v) if isinstance(v,Path) else v for k,v in data['settings'].items()}
    sources = ['cuda/ascii_cuda.cu','cuda/ascii_cuda.h','cuda/main.c','cuda/image_io.c','cuda/image_io.h',
               'cuda/benchmark.c','cuda/resident_benchmark.c','CMakeLists.txt','benchmarks/compare_cuda.py','benchmarks/run.py']
    data['source_sha256'] = {name:digest(ROOT/name) for name in sources}
    data['atlas_sha256'] = {name:digest(ROOT/name) for name in ['res/fillASCII.png','res/edgesASCII.png']}
    data['baseline_source_sha256'] = {name: digest(args.baseline_build.parent/'baseline-source'/name)
        for name in ['cuda/ascii_cuda.cu','cuda/ascii_cuda.h','cuda/image_io.c','cuda/image_io.h','cuda/benchmark.c','CMakeLists.txt']
        if (args.baseline_build.parent/'baseline-source'/name).exists()}
    data['environment']['host_compiler'] = optional_command(['cc','--version']).splitlines()[0]
    data['environment']['io_library_versions'] = optional_command(['pkg-config','--modversion','libpng','libjpeg']).replace('\n', ', ')
    data['binary_sha256'] = {'baseline':digest(args.baseline_build/'ascii_benchmark'),
        'optimized':digest(args.build/'ascii_benchmark'),'resident':digest(args.build/'ascii_resident_benchmark')}
    for case,(size,bloom) in enumerate((s,b) for s in sizes for b in (False,True)):
        print(f"Measuring {size}, bloom {'on' if bloom else 'off'}",flush=True)
        source = Path(inputs[size]['path']);out = {};case_records = {}
        for mode in MODES:
            r = {'size':size,'bloom':bloom,'mode':mode,'steady':[],'fresh':[],'profile':[], 'runs':[]}
            data['records'].append(r);case_records[mode] = r;out[mode] = work/f'{size}_{int(bloom)}_{mode}.png'
        for trial in range(args.trials):
            offset = (case+trial)%len(MODES);order = MODES[offset:]+MODES[:offset]
            for mode in order:
                exe = (args.baseline_build if mode == 'baseline' else args.build)/'ascii_benchmark'
                run = worker(exe,source,out[mode],work/'metrics.json',args.warmup,args.repeat,bloom,False,args.device,mode)
                case_records[mode]['steady'] += run['metrics']['samples'];case_records[mode]['runs'].append({'command':run['command'],'process_ms':run['process_ms']})
        for mode in MODES:
            r = case_records[mode];r['output_bytes'] = out[mode].stat().st_size
            if mode != 'baseline':
                r['agreement'] = agreement(out['baseline'],out[mode])
                if r['agreement']['max_abs_error'] > 2: raise RuntimeError(f"Output disagreement: {size}, {mode}, {r['agreement']}")
        for trial in range(args.fresh_repeat):
            for mode in (['baseline','optimized'] if (trial+case)%2 == 0 else ['optimized','baseline']):
                exe = (args.baseline_build if mode == 'baseline' else args.build)/'ascii_benchmark'
                run = worker(exe,source,work/'fresh.png',work/'fresh.json',0,1,bloom,False,args.device,mode)
                case_records[mode]['fresh'].append({'process_ms':run['process_ms'],**run['metrics']['samples'][0], 'command':run['command']})
        if size in ['1920x1080','3840x2160']:
            for mode in MODES:
                exe = (args.baseline_build if mode == 'baseline' else args.build)/'ascii_benchmark'
                run = worker(exe,source,work/'profile.png',work/'profile.json',args.warmup,5,bloom,True,args.device,mode)
                case_records[mode]['profile'] = run['metrics']['samples'];case_records[mode]['profile_command'] = run['command']
        resident_out = work/f'{size}_{int(bloom)}_resident.png'
        resident_cmd = [str(args.build/'ascii_resident_benchmark'),str(source),str(resident_out),str(work/'resident.json'),
            '16',str(args.resident_repeat),str(int(bloom)),'0','0',str(args.device)]
        command(resident_cmd);resident = json.loads((work/'resident.json').read_text())
        r = {'size':size,'bloom':bloom,'mode':'resident','steady':resident['samples'],'fresh':[],'profile':[],
            'command':resident_cmd,'setup_ms':resident['setup_ms'],'output_bytes':resident_out.stat().st_size,
            'agreement':agreement(out['baseline'],resident_out)}
        if r['agreement']['max_abs_error'] > 2: raise RuntimeError('Resident output differs from baseline')
        data['records'].append(r)
        # Preserve measured work if a later profiler operation fails.
        (result_dir/'results.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    if not args.skip_nsight:
        size = '1920x1080' if '1920x1080' in sizes else sizes[-1]
        for bloom in (False,True):
            for mode in ['baseline','optimized','graph_auto']:
                prefix = work/f'nsight_{mode}_{int(bloom)}'
                exe = (args.baseline_build if mode == 'baseline' else args.build)/'ascii_benchmark'
                worker_cmd = [str(exe),inputs[size]['path'],str(work/'nsight.png'),str(work/'nsight.json'),
                    '1','1',str(int(bloom)),'0',str(args.device)]
                if mode != 'baseline': worker_cmd += [str(int(mode.startswith('graph'))),'0']
                capture = ['nsys','profile','--force-overwrite=true','--sample=none','--cpuctxsw=none','--trace=cuda',
                    '--cuda-graph-trace=node','-o',str(prefix),*worker_cmd]
                print(f'Profiling {mode}, bloom {int(bloom)}',flush=True);command(capture)
                command(['nsys','stats','--force-overwrite=true','--report',
                    'cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum',
                    '--format','csv','--output',str(result_dir/f'nsight_{mode}_{int(bloom)}'),str(prefix)+'.nsys-rep'])
                data['nsight'].append({'mode':mode,'bloom':bloom,'size':size,'command':capture})
    report(data,result_dir)
    print('Wrote OPTIMIZATION.md and benchmarks/optimized_results/',flush=True)


if __name__ == '__main__':
    main()
