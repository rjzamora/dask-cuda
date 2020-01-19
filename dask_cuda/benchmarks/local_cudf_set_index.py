import argparse
import math
from collections import defaultdict
from time import perf_counter as clock

from dask.base import tokenize
from dask.dataframe.core import new_dd_object
from dask.distributed import Client, performance_report, wait
from dask.utils import format_bytes, format_time, parse_bytes
from dask_cuda import LocalCUDACluster
from dask_cuda.initialize import initialize

import cudf
import cupy
import numpy

# Benchmarking dask-based set_index operation (sorting)


def generate_chunk(i_chunk, local_size, num_chunks):
    # Setting a seed that triggers max amount of comm in the two-GPU case.
    cupy.random.seed(42 * i_chunk)

    # "key" column is a random uniform selection of global indices
    #
    # "payload" column is a random permutation of the chunk_size

    df = cudf.DataFrame(
        {
            # Note: Can also use other distributions for non-uniform
            #       sorted partitions...
            "key": cupy.random.randint(
                low=0, high=local_size*num_chunks, size=local_size, dtype="int64"
            ),
            "payload": cupy.random.permutation(
                cupy.arange(local_size, dtype="int64")
            ),
        }
    )

    return df


def get_random_ddf(chunk_size, num_chunks, args):

    parts = [chunk_size for i in range(num_chunks)]
    meta = generate_chunk(0, 4, 1)
    divisions = [None] * (len(parts) + 1)

    name = "generate-data-" + tokenize(chunk_size, num_chunks)

    graph = {
        (name, i): (generate_chunk, i, part, len(parts))
        for i, part in enumerate(parts)
    }

    return new_dd_object(graph, name, meta, divisions)


def run(args, write_profile=None):
    # Generate random Dask dataframe
    ddf_base = get_random_ddf(args.chunk_size, args.n_workers, args).persist()
    wait(ddf_base)

    assert(len(ddf_base.dtypes) == 2)
    data_processed = len(ddf_base) * sum([t.itemsize for t in ddf_base.dtypes])

    # If desired, calculate divisions separtately
    if args.known_divisions:
        divisions = ddf_base['key']._repartition_quantiles(
            args.n_workers, upsample=1.0
        ).compute().to_list()
    else:
        divisions = None

    # Lazy set_index operation
    ddf_sorted = ddf_base.set_index(
        "key",
        npartitions=ddf_base.npartitions,
        divisions=divisions,
    )

    # Execute the operations to benchmark
    if write_profile is not None:
        with performance_report(filename=args.profile):
            t1 = clock()
            wait(ddf_sorted.persist())
            took = clock() - t1
    else:
        t1 = clock()
        wait(ddf_sorted.persist())
        took = clock() - t1
    return (data_processed, took)


def main(args):
    # Set up workers on the local machine
    if args.protocol == "tcp":
        cluster = LocalCUDACluster(
            protocol=args.protocol,
            n_workers=args.n_workers,
            CUDA_VISIBLE_DEVICES=args.devs,
        )
    else:
        enable_infiniband = args.enable_infiniband
        enable_nvlink = args.enable_nvlink
        enable_tcp_over_ucx = args.enable_tcp_over_ucx
        cluster = LocalCUDACluster(
            protocol=args.protocol,
            n_workers=args.n_workers,
            CUDA_VISIBLE_DEVICES=args.devs,
            ucx_net_devices="auto",
            enable_tcp_over_ucx=enable_tcp_over_ucx,
            enable_infiniband=enable_infiniband,
            enable_nvlink=enable_nvlink,
        )
        initialize(
            create_cuda_context=True,
            enable_tcp_over_ucx=enable_tcp_over_ucx,
            enable_infiniband=enable_infiniband,
            enable_nvlink=enable_nvlink,
        )
    client = Client(cluster)

    def _worker_setup():
        import rmm
        rmm.reinitialize(pool_allocator=not args.no_rmm_pool, devices=0)
        cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

    client.run(_worker_setup)

    took_list = []
    for _ in range(args.runs - 1):
        took_list.append(run(args, write_profile=None))
    took_list.append(
        run(args, write_profile=args.profile)
    )  # Only profiling the last run

    # Collect, aggregate, and print peer-to-peer bandwidths
    incoming_logs = client.run(lambda dask_worker: dask_worker.incoming_transfer_log)
    bandwidths = defaultdict(list)
    total_nbytes = defaultdict(list)
    for k, L in incoming_logs.items():
        for d in L:
            if d["total"] >= args.ignore_size:
                bandwidths[k, d["who"]].append(d["bandwidth"])
                total_nbytes[k, d["who"]].append(d["total"])
    bandwidths = {
        (cluster.scheduler.workers[w1].name, cluster.scheduler.workers[w2].name): [
            "%s/s" % format_bytes(x) for x in numpy.quantile(v, [0.25, 0.50, 0.75])
        ]
        for (w1, w2), v in bandwidths.items()
    }
    total_nbytes = {
        (
            cluster.scheduler.workers[w1].name,
            cluster.scheduler.workers[w2].name,
        ): format_bytes(sum(nb))
        for (w1, w2), nb in total_nbytes.items()
    }

    if args.markdown:
        print("```")
    print("Sort (set_index) benchmark")
    print("-------------------------------")
    print(f"Chunk-size  | {args.chunk_size}")
    print(f"Divisions   | {args.known_divisions}")
    print(f"Ignore-size | {format_bytes(args.ignore_size)}")
    print(f"Protocol    | {args.protocol}")
    print(f"Device(s)   | {args.devs}")
    print(f"rmm-pool    | {(not args.no_rmm_pool)}")
    if args.protocol == "ucx":
        print(f"tcp         | {args.enable_tcp_over_ucx}")
        print(f"ib          | {args.enable_infiniband}")
        print(f"nvlink      | {args.enable_nvlink}")
    print("===============================")
    print("Wall-clock  | Throughput")
    print("-------------------------------")
    for data_processed, took in took_list:
        throughput = int(data_processed / took)
        print(f"{format_time(took)}      | {format_bytes(throughput)}/s")
    print("===============================")
    if args.markdown:
        print(
            "\n```\n<details>\n<summary>Worker-Worker Transfer Rates</summary>\n\n```"
        )
    print("(w1,w2)     | 25% 50% 75% (total nbytes)")
    print("-------------------------------")
    for (d1, d2), bw in sorted(bandwidths.items()):
        print(
            "(%02d,%02d)     | %s %s %s (%s)"
            % (d1, d2, bw[0], bw[1], bw[2], total_nbytes[(d1, d2)])
        )
    if args.markdown:
        print("```\n</details>\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge (dask/cudf) on LocalCUDACluster benchmark"
    )
    parser.add_argument(
        "-d", "--devs", default="0", type=str, help='GPU devices to use (default "0").'
    )
    parser.add_argument(
        "-p",
        "--protocol",
        choices=["tcp", "ucx"],
        default="tcp",
        type=str,
        help="The communication protocol to use.",
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        default=1_000_000,
        metavar="n",
        type=int,
        help="Chunk size (default 1_000_000)",
    )
    parser.add_argument(
        "--ignore-size",
        default="1 MiB",
        metavar="nbytes",
        type=parse_bytes,
        help='Ignore messages smaller than this (default "1 MB")',
    )
    parser.add_argument(
        "--no-rmm-pool", action="store_true", help="Disable the RMM memory pool"
    )
    parser.add_argument(
        "--profile",
        metavar="PATH",
        default=None,
        type=str,
        help="Write dask profile report (E.g. dask-report.html)",
    )
    parser.add_argument(
        "--markdown", action="store_true", help="Write output as markdown"
    )
    parser.add_argument("--runs", default=3, type=int, help="Number of runs")
    parser.add_argument(
        "--enable-tcp-over-ucx",
        action="store_true",
        dest="enable_tcp_over_ucx",
        help="Enable tcp over ucx.",
    )
    parser.add_argument(
        "--enable-infiniband",
        action="store_true",
        dest="enable_infiniband",
        help="Enable infiniband over ucx.",
    )
    parser.add_argument(
        "--enable-nvlink",
        action="store_true",
        dest="enable_nvlink",
        help="Enable NVLink over ucx.",
    )
    parser.add_argument(
        "--disable-tcp-over-ucx",
        action="store_false",
        dest="enable_tcp_over_ucx",
        help="Disable tcp over ucx.",
    )
    parser.add_argument(
        "--disable-infiniband",
        action="store_false",
        dest="enable_infiniband",
        help="Disable infiniband over ucx.",
    )
    parser.add_argument(
        "--disable-nvlink",
        action="store_false",
        dest="enable_nvlink",
        help="Disable NVLink over ucx.",
    )
    parser.set_defaults(
        enable_tcp_over_ucx=True, enable_infiniband=True, enable_nvlink=True
    )
    parser.add_argument(
        "--known-divisions",
        action="store_true",
        help="Calculate divisions before set_index operation",
    )
    args = parser.parse_args()
    args.n_workers = len(args.devs.split(","))
    return args


if __name__ == "__main__":
    main(parse_args())
