import csv
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Модель Hugging Face
MODEL_NAME = "khanfs/ChemSolubilityBERTa"

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "chemistry_results.csv"
CHART_FILE = BASE_DIR / "chemistry_chart.png"

tokenizer = None
model = None

# Речовини для початкового аналізу
molecules = [
    ("Етанол", "CCO"),
    ("Оцтова кислота", "CC(=O)O"),
    ("Бензен", "c1ccccc1"),
    ("Ацетон", "CC(=O)C"),
    ("Метанол", "CO"),
    ("Пропанол", "CCCO"),
    ("Фенол", "Oc1ccccc1"),
    ("Толуен", "Cc1ccccc1"),
    ("Етилацетат", "CCOC(=O)C"),
    ("Гліцерин", "OCC(O)CO"),
    ("Піридин", "c1ccncc1"),
    ("Циклогексан", "C1CCCCC1")
]
results = []


# Завантаження моделі
def load_model():
    global tokenizer, model
    if model is None:
        print("\nЗавантаження моделі Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )
        model.eval()
        print("Модель готова до роботи!")
    return model


# Перевірка молекули
def check_molecule(smiles):
    if not smiles.strip():
        raise ValueError("SMILES не може бути порожнім!")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Некоректний SMILES!")
    formula = rdMolDescriptors.CalcMolFormula(molecule)
    canonical_smiles = Chem.MolToSmiles(molecule)
    return formula, canonical_smiles


# Прогноз розчинності
def predict_solubility(smiles):
    load_model()
    inputs = tokenizer(
        smiles,
        return_tensors="pt",
        truncation=False
    )
    if inputs["input_ids"].shape[1] > 512:
        raise ValueError("SMILES занадто довгий для моделі!")
    with torch.no_grad():
        output = model(**inputs)
    # Прогноз logS (log10 моль/л)
    value = output.logits.item()
    return round(value, 4)


# Збереження результатів у CSV
def save_results():
    columns = [
        "molecule_name",
        "formula",
        "smiles",
        "predicted_property",
        "predicted_value"
    ]
    with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    print("\nРезультати збережено у chemistry_results.csv")


# Завантаження раніше отриманих результатів
def load_results():
    if CSV_FILE.exists():
        with open(CSV_FILE, "r", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    row["predicted_value"] = float(row["predicted_value"])
                    results.append(row)
                except (ValueError, KeyError, TypeError):
                    continue


# Аналіз однієї речовини
def analyze_molecule(name, smiles):
    if not name.strip():
        raise ValueError("Назва речовини не може бути порожньою!")
    formula, smiles = check_molecule(smiles)
    value = predict_solubility(smiles)
    result = {
        "molecule_name": name,
        "formula": formula,
        "smiles": smiles,
        "predicted_property": "Water solubility (logS)",
        "predicted_value": value
    }
    # Оновлюємо запис, якщо речовина вже існує
    for i, old in enumerate(results):
        if old["smiles"] == smiles:
            results[i] = result
            break
    else:
        results.append(result)
    print("\n     AI RESULT")
    print("Речовина:", name)
    print("Формула:", formula)
    print("SMILES:", smiles)
    print("Прогнозована властивість: розчинність у воді")
    print(f"Прогноз logS: {value:.4f}")
    save_results()


# Додавання власної речовини
def add_molecule():
    print("\n     NEW MOLECULE")
    name = input("Назва речовини: ").strip()
    smiles = input("SMILES: ").strip()
    analyze_molecule(name, smiles)


# Аналіз 12 речовин
def analyze_all():
    print("\n     MOLECULE ANALYSIS")
    successful = 0
    for name, smiles in molecules:
        try:
            analyze_molecule(name, smiles)
            successful += 1
        except Exception as error:
            print(f"\nПомилка для {name}: {error}")
    print("\n     REPORT")
    print("Усього речовин:", len(molecules))
    print("Успішно проаналізовано:", successful)
    print("Помилок:", len(molecules) - successful)


# Перегляд результатів
def show_results():
    if not results:
        print("\nРезультатів поки немає.")
        return
    print("\n     RESULTS")
    for i, item in enumerate(results, 1):
        print(
            f"{i}. {item['molecule_name']} | "
            f"{item['formula']} | "
            f"logS: {item['predicted_value']:.4f}"
        )


# Сортування за розчинністю
def sort_results():
    if not results:
        print("\nСпочатку проаналізуйте речовини!")
        return
    print("\n1. Від найбільшої розчинності")
    print("2. Від найменшої розчинності")
    choice = input("Ваш вибір: ")
    if choice not in ("1", "2"):
        print("Невірний вибір!")
        return
    sorted_items = sorted(
        results,
        key=lambda x: x["predicted_value"],
        reverse=(choice == "1")
    )
    print("\n     SORTED RESULTS")
    for item in sorted_items:
        print(
            f"{item['molecule_name']}: "
            f"{item['predicted_value']:.4f}"
        )


# Пошук речовини
def search_molecule():
    name = input("\nВведіть назву речовини: ").strip().lower()
    if not name:
        print("Введіть назву для пошуку!")
        return
    found = [
        item for item in results
        if name == item["molecule_name"].lower()
    ]
    if not found:
        print("Речовину не знайдено!")
        return
    for item in found:
        print("\nНазва:", item["molecule_name"])
        print("Формула:", item["formula"])
        print("SMILES:", item["smiles"])
        print("Прогноз logS:", item["predicted_value"])


# Порівняння декількох речовин
def compare_molecules():
    if len(results) < 2:
        print("\nДля порівняння потрібно мінімум 2 речовини.")
        return
    show_results()
    choice = input(
        "\nВведіть номери речовин через кому (наприклад, 1,2,3): "
    )
    try:
        numbers = [int(x.strip()) for x in choice.split(",")]
        if len(set(numbers)) < 2:
            print("Виберіть мінімум дві різні речовини!")
            return
        selected = [results[i - 1] for i in numbers]
        print("\n     COMPARISON")
        for item in selected:
            print(
                f"{item['molecule_name']}: "
                f"logS = {item['predicted_value']:.4f}"
            )
        best = max(selected, key=lambda x: x["predicted_value"])
        print("\nНайбільша прогнозована розчинність:")
        print(best["molecule_name"])
    except (ValueError, IndexError):
        print("Некоректні номери речовин!")


# Побудова графіка
def create_chart():
    if not results:
        print("\nНемає результатів для графіка!")
        return
    names = [item["molecule_name"] for item in results]
    values = [item["predicted_value"] for item in results]
    plt.figure(figsize=(11, 6))
    plt.barh(names, values)
    plt.xlabel("Прогнозована розчинність (logS)")
    plt.title("AI Chemistry Lab — порівняння молекул")
    plt.axvline(0, color="gray", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150)
    plt.show()
    plt.close()
    print("\nГрафік збережено:", CHART_FILE)


# Порівняння з експериментальним значенням
def compare_experiment():
    if not results:
        print("\nСпочатку проаналізуйте речовини!")
        return
    show_results()
    try:
        number = int(input("\nНомер речовини: "))
        item = results[number - 1]
        if number < 1:
            raise IndexError
        value = float(
            input("Експериментальне logS (з джерела): ").replace(",", ".")
        )
        source = input("Джерело або DOI: ").strip()
        if not source:
            print("Необхідно вказати джерело!")
            return
        difference = abs(item["predicted_value"] - value)
        print("\n     EXPERIMENT COMPARISON")
        print("Речовина:", item["molecule_name"])
        print("Прогноз:", item["predicted_value"])
        print("Експеримент:", value)
        print(f"Абсолютна різниця: {difference:.4f}")
        print("Джерело:", source)
    except (ValueError, IndexError):
        print("Помилка введення даних!")


# Інформація про модель
def model_info():
    print("\n      MODEL INFORMATION ")
    print("Модель:", MODEL_NAME)
    print("Hugging Face:", "https://huggingface.co/" + MODEL_NAME)
    print("Вхід: SMILES-представлення молекули")
    print("Вихід: прогноз розчинності у воді (logS)")
    print("Тип: навчена модель регресії")
    print("Дані навчання: ESOL")
    print("Прогнози не є експериментальними вимірюваннями.")


# Головне меню
def show_menu():
    print("\n           AI CHEMISTRY LAB")
    print("1. Додати та проаналізувати речовину")
    print("2. Проаналізувати 12 речовин")
    print("3. Переглянути результати")
    print("4. Сортувати за розчинністю")
    print("5. Пошук речовини")
    print("6. Порівняти речовини")
    print("7. Побудувати графік")
    print("8. Порівняти з експериментом")
    print("9. Інформація про модель")
    print("0. Вихід")


# Запуск програми
def main():
    load_results()
    while True:
        show_menu()
        choice = input("\nОберіть дію: ")
        try:
            if choice == "1":
                add_molecule()
            elif choice == "2":
                analyze_all()
            elif choice == "3":
                show_results()
            elif choice == "4":
                sort_results()
            elif choice == "5":
                search_molecule()
            elif choice == "6":
                compare_molecules()
            elif choice == "7":
                create_chart()
            elif choice == "8":
                compare_experiment()
            elif choice == "9":
                model_info()
            elif choice == "0":
                print("\nДо побачення!")
                break
            else:
                print("Невірний вибір!")
        except Exception as error:
            print("\nПомилка:", error)


if __name__ == "__main__":
    main()
