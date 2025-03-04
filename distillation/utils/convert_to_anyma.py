import argparse
import torch
import pickle as pkl

def main(input_ckpt, output_pkl):
    w = torch.load(input_ckpt)
    w_student = {k: v for k, v in w['state_dict'].items() if 'student'}
    
    weights = {}

    for k, v in w_student.items():
        new_k = k.replace('student.model.model.', 'backbone.')
        weights[new_k] = v.detach().cpu().numpy()

    # Save in Detectron2 compatible format with metadata
    res = {
        "model": weights,
        "__author__": "dinov2_distilled",
        "matching_heuristics": True
    }

    with open(output_pkl, "wb") as f:
        pkl.dump(res, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert student model weights to Anyma format.")
    parser.add_argument("input_ckpt", type=str, help="Path to the input checkpoint file.")
    parser.add_argument("output_pkl", type=str, help="Path to the output pickle file.")
    args = parser.parse_args()
    
    main(args.input_ckpt, args.output_pkl)