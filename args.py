import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--number', type=int, default=10, help="number pls")

FLAGS, unparsed = parser.parse_known_args()

print(FLAGS.number**2)

