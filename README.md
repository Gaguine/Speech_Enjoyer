# Argparse CLI Cheatsheet for Emotion Analysis and Visualization

This cheatsheet provides an overview of the command-line interface (CLI) options for the emotion analysis and visualization script using `argparse`.

---

## **Available Commands**

### 1. `analyze`

Analyze emotions and semantics in a text file and save the results to a CSV file.

#### **Usage**

```
python main.py analyze -i <input_file> -o <output_file>
```

#### **Arguments**

- `**-i**`**,** `**--input**` (required): Path to the input text file.
    
    - Example: `speech/Elon Musk Speech`
        
- `**-o**`**,** `**--output**` (optional): Path to save the output CSV file (default: `output.csv`).
    
    - Example: `results/output.csv`
        

#### **Example**

```
python main.py analyze -i "speech/Elon Musk Speech" -o "output.csv"
```

---

### 2. `visualize`

Visualize emotion data from a CSV file.

#### **Usage**

```
python main.py visualize -i <input_file> -t <type> [-s <start>] [-e <end>]
```

#### **Arguments**

- `**-i**`**,** `**--input**` (required): Path to the input dataset (CSV file).
    
    - Example: `output.csv`
        
- `**-t**`**,** `**--type**` (required): Type of visualization to generate.
    
    - Options:
        
        - `grouped`: Grouped bar chart.
            
        - `stacked`: Stacked bar chart.
            
        - `semantic`: Line chart of semantic dynamics.
            
    - Example: `grouped`
        
- `**-s**`**,** `**--start**` (optional): Start index for the grouped bar chart (default: `0`).
    
    - Example: `0`
        
- `**-e**`**,** `**--end**` (optional): End index for the grouped bar chart (default: `10`).
    
    - Example: `15`
        

#### **Examples**

- Visualize a **grouped bar chart** for phrases 0 to 15:
    
    ```
    python main.py visualize -i "output.csv" -t grouped -s 0 -e 15
    ```
    
- Visualize a **stacked bar chart**:
    
    ```
    python main.py visualize -i "output.csv" -t stacked
    ```
    
- Visualize **semantic dynamics**:
    
    ```
    python main.py visualize -i "output.csv" -t semantic
    ```
    

---

## **Command Summary**

|Command|Arguments|Description|
|---|---|---|
|`analyze`|`-i <input_file>`, `-o <output_file>`|Analyze emotions and semantics in text.|
|`visualize`|`-i <input_file>`, `-t <type>`, `-s <start>`, `-e <end>`|Visualize emotion data from CSV.|

---

## **Additional Notes**

- Ensure the required modules (`pandas`, `matplotlib`, etc.) are installed before running the script.
    
- Use valid file paths for both input and output to avoid errors.
    
- The `visualize` command supports flexible visualization types to meet different analysis needs.

# Acknowledgements
Models used:
- https://huggingface.co/tabularisai/multilingual-sentiment-analysis (sentiment analysis)
- https://huggingface.co/j-hartmann/emotion-english-distilroberta-base (emotion)\
- https://huggingface.co/dslim/distilbert-NER (ner analysis)
