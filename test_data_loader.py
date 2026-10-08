from experiments.data_loader import load_dataset

BASE_PATH = r'C:\Users\Usuario UTP\Documents\PANDA\data\MADatasets\MADatasets'
# BASE_PATH = r'C:\data\MADatasets\MADatasets'

for name in ["Breast", "Iris", "spirals", "Voice"]:
    print(f"\n{'='*50}\n{name}")

    try:
        datasets = load_dataset(name, BASE_PATH)

        for data in datasets:
            train_data, val_data = data["train_data"], data["val_data"]

            print("name:", data["name"])
            print("input_dim:", data["input_dim"])
            print("K:", data["K"], "T:", data["T"])
            print("X_train:", train_data[0].shape)
            print("Y_train:", train_data[1].shape)
            print("M_train:", train_data[2].shape)
            print("Z_train:", train_data[3].shape)
            print("X_val:", val_data[0].shape)
            print("Y_val:", val_data[1].shape)

    except Exception as e:
        print("ERROR:", type(e).__name__, e)