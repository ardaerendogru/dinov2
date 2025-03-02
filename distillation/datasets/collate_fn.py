import torch

def collate_data_and_cast(samples_list):
    n_global_crops = len(samples_list[0]["global_crops"])
    collated_global_crops = torch.stack([s["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    return {
        "collated_global_crops": collated_global_crops,
        "collated_local_crops": None,
    }