import json

def score(input_jsonl, output_errors=None):

    hit = 0
    count = 0
    errors = []
    with open(input_jsonl) as f:
        for line in f:
            obj = json.loads(line)
            expected = obj['expected']
            actual = obj['actual']

            if actual == expected:
                hit += 1
            else:
                errors.append(obj)
            count += 1

    print(f'Accuracy = {hit}/{count} = {hit / count}')
    if output_errors is not None:
        with open(output_errors, 'w') as o:
            for obj in errors:
                o.write(json.dumps(obj) + '\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input JSONL file with predictions ("expected" + "actual" keys in each record)')
    parser.add_argument('--errors', '-e', help='Save errors in this JSONL file (for further analysis)')

    args = parser.parse_args()
    score(args.input, args.errors)

