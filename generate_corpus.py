import random
import argparse
import json

PAIRS = [("(",")"), ("[","]")]

def generate_dyck(length: int, max_depth: int, k: int = 2):
    """Return a random Dyck string of exactly `length` tokens (even),
    with maximum nesting depth <= max_depth, or None on failure."""
    assert length % 2 == 0
    stack, result, depth = [], [], 0
    remaining = length
    max_depth_reached = 0

    for step in range(length):
        remaining = length - step
        must_close = (len(stack) == remaining)
        can_open   = (depth < max_depth) and (remaining > len(stack) + 1)

        choices = []
        if can_open and not must_close:
            choices.append("open")
        if stack:
            choices.append("close")
        
        if not choices:
            return None
        
        match random.choice(choices):
            case "open":
                pair = random.choice(PAIRS[:k])
                stack.append(pair)
                result.append(pair[0])
                depth += 1
                max_depth_reached = max(max_depth_reached, depth)
            case "close":
                pair = stack.pop()
                result.append(pair[1])
                depth = len(stack)
    
    res = "".join(result) if not stack else None

    return res, max_depth_reached


def insert_error(dyck: str, k: int = 2):
    errors = ["e1", "e2", "e3", "e4"]
    error = random.choice(errors)
    pairs = PAIRS[:k]
    closings = [p[1] for p in pairs]
    openers  = [p[0] for p in pairs]
    closer_to_opener = {p[1]: p[0] for p in pairs}

    if error == "e1": # missing closer
        idx = random.choice([i for i, c in enumerate(dyck) if c in closings])
        e_tok = dyck[idx]
        res = dyck[:idx] + dyck[idx+1:]
    
    if error == "e2": # spurious opener
        idx = random.randrange(len(dyck) + 1)
        opener = random.choice(openers)
        e_tok = opener
        res = dyck[:idx] + opener + dyck[idx:]
    
    if error == "e3": # type mismatch
        idx = random.choice([i for i, c in enumerate(dyck) if c in closings])
        replacer = random.choice([c for c in closings if c != dyck[idx]])
        e_tok = replacer
        res = dyck[:idx] + replacer + dyck[idx+1:]
    
    if error == "e4":  # premature close
        existing_closers = [c for c in closings if closer_to_opener[c] in dyck]
        closer = random.choice(existing_closers)
        opener = closer_to_opener[closer]
        first = dyck.find(opener)
        idx = random.randrange(0, first + 1)
        e_tok = closer
        res = dyck[:idx] + closer + dyck[idx:]
    
    return res, error, idx, e_tok



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-length", type = int, default = 4)
    parser.add_argument("--max-length", type = int, default = 40)
    parser.add_argument("--max-depth",  type = int, default = 4)
    parser.add_argument("--size",       type = int, default = 5_000)
    parser.add_argument("--k",          type = int, default = 2)
    parser.add_argument("--seed",       type = int, default = 42)
    parser.add_argument("--output",     type = str, default = "dyck.jsonl")
    args = parser.parse_args()
    
    random.seed(args.seed)

    with open(args.output, "w") as f:
        for i in range(args.size):
            result = None
            while result is None:
                length = random.randrange(args.min_length, args.max_length + 1, 2)
                result = generate_dyck(length, args.max_depth, args.k)
            original, depth = result

            if random.random() < 0.5:
                is_corrupted = True
                actual, error, e_pos, e_tok = insert_error(original, args.k)
            else:
                is_corrupted = False
                actual, error, e_pos, e_tok = original, None, None, None
            
            record = {
                "id":             i,
                "original":       original,
                "actual":         actual,
                "is_corrupted":   is_corrupted,
                "error_type":     error,
                "error_position": e_pos,
                "error_token":    e_tok,
                "depth":          depth,
                "length":         len(actual),
            }
            f.write(json.dumps(record) + "\n")
        

        


