"""
Runs a bunch of parallel workers
Reads pickled files as input
Pickles results to an output file
"""
import sys
import argparse
import six

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--input-file',
        type=str,
        help='a pickle file',
        default='input.pkl')
    parser.add_argument('--output-file',
        type=str,
        help='a pickle file',
        default='output.pkl')

    parser.set_defaults()
    return parser.parse_args()

def main(args=sys.argv[1:]):
    args = parse_args()
    with open(args.input_file, "rb") as input_file:
        batched_workers = six.moves.cPickle.load(input_file)

    results = []
    for worker in batched_workers.workers:
        results.append(worker.run(batched_workers.shared_obj))

    with open(args.output_file, "wb") as output_file:
        six.moves.cPickle.dump(results, output_file, protocol=2)

if __name__ == "__main__":
    main(sys.argv[1:])
