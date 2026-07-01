
from hoi.evaluation_tools.train_tools_visual_force_prediction import DinoClipForceNet, ForceDataset, train_force_prior, predict_force_from_folder, compute_class_priors
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



if __name__ == "__main__":

    PRIORS, GLOBAL_MEAN = compute_class_priors("/data/robot_tests/dataset")

    #4, 15

    # train_force_prior("/data/robot_tests/dataset", alpha=1.0, epochs=30, 
    #                   save_path="/data/robot_tests/models/forcenet_forcetotal_alpha1.pt",
    #                   PRIORS=PRIORS, GLOBAL_MEAN=GLOBAL_MEAN)
    
    model = DinoClipForceNet(
        alpha=1,
    )

    state = torch.load("/data/robot_tests/models/forcenet_forcetotal_alpha1.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    predictions = []
    for i in range(5):
        f = predict_force_from_folder(
        model,
        f"/data/robot_tests/eval/drawer_{i}",
        PRIORS=PRIORS,
        GLOBAL_MEAN=GLOBAL_MEAN,
    )
        print(f)
        predictions.append(f)
    print("Final predictions:", predictions)
    print("Average prediction:", sum(predictions)/len(predictions))


# alpha 1 is gut