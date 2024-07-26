import csv
import os


def export_results_to_csv(results, model_type, dataset_type, output_folder):
    filename = f"{output_folder}/{model_type}_{dataset_type}_results.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Set",
                "Steps",
                "Position MSE Mean",
                "Position MSE Std",
                "Velocity MSE Mean",
                "Velocity MSE Std",
            ]
        )

        for key, value in results.items():
            split, steps = key.split("_")
            pos_mean, pos_std = value["position"]
            vel_mean, vel_std = value["velocity"]
            writer.writerow([split, steps, pos_mean, pos_std, vel_mean, vel_std])


def export_consolidated_results(all_results, output_folder):
    filename = f"{output_folder}/consolidated_results.csv"

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Problem",
                "Model",
                "Set",
                "Steps",
                "Position MSE Mean",
                "Position MSE Std",
                "Velocity MSE Mean",
                "Velocity MSE Std",
            ]
        )

        for problem in all_results:
            for model in all_results[problem]:
                results = all_results[problem][model]
                for key, value in results.items():
                    split, steps = key.split("_")
                    pos_mean, pos_std = value["position"]
                    vel_mean, vel_std = value["velocity"]
                    writer.writerow(
                        [
                            problem,
                            model,
                            split,
                            steps,
                            pos_mean,
                            pos_std,
                            vel_mean,
                            vel_std,
                        ]
                    )
