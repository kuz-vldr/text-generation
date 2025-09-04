import os
from sklearn.model_selection import train_test_split

def split_dataset(
        text: list[str],
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1) -> dict:
    """
    Разделение датасета на три выборки: тренировучную, валидационную и тестовую и сохранение их в файл.
    """
    assert (train + val + test == 1.0)
    assert (train * val * test > 0.0)

    os.makedirs("data", exist_ok=True)
    
    train_set, temp_set = train_test_split(
        text, test_size=(val + test), train_size=train, random_state=42)


    val_set, test_set = train_test_split(
        temp_set, test_size=test/(val + test), train_size=val/(val + test), 
        random_state=42)


    with open(os.path.join("data", "train.txt"), "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in train_set)
    
    with open(os.path.join("data", "val.txt"), "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in val_set)
    
    with open(os.path.join("data", "test.txt"), "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in test_set)

    print(f"train set len:       {len(train_set)}")
    print(f"validation set len:  {len(val_set)}")
    print(f"test set len:        {len(test_set)}")
    print(f"Files saved to: data")

    return {"train": train_set, "val": val_set, "test": test_set}

