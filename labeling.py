import json
import time
import google.generativeai as palm
import os

# Configure Google Generative AI
palm.configure(api_key="AIzaSyC25fzvcu630l5MG2VNwolZpRk94ZJhRlc")  # Replace with your actual API key

# Predefined annotation categories
ANNOTATION_CATEGORIES = [
    "Deep Learning",
    "Computer Vision",
    "Reinforcement Learning",
    "NLP",
    "Optimization"
]

def get_annotation_from_gemini(title, abstract):
    """
    Classify a research paper into one of the predefined categories using the Gemini API.
    """
    prompt = (
        "Please classify the following research paper into one of the following categories: "
        "Deep Learning, Computer Vision, Reinforcement Learning, NLP, Optimization.\n\n"
        f"Title: {title}\n\nAbstract: {abstract}\n\n"
        "Provide only the category as your answer."
    )
    max_retries = 5
    retry_delay = 30

    for attempt in range(max_retries):
        try:
            model = palm.GenerativeModel(model_name="gemini-2.0-flash")
            response = model.generate_content(prompt)

            if not response or not response.text:
                print(f"Warning: Empty response received for title: {title}")
                return "Unclassified"

            label = response.text.strip()

            if label not in ANNOTATION_CATEGORIES:
                for cat in ANNOTATION_CATEGORIES:
                    if cat.lower() in label.lower():
                        label = cat
                        break
                else:
                    label = "Unclassified"
            return label
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if "429" in str(e):
                print(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return "Error"
    return "Error"

def classify_papers_in_json(json_file_path):
    """
    Read a JSON file, classify each paper, and append the label to the same file.
    """
    try:
        if not os.path.exists(json_file_path):
            print(f"Error: File '{json_file_path}' not found.")
            return
        
        # Load JSON data
        with open(json_file_path, "r", encoding="utf-8") as file:
            try:
                papers = json.load(file)
                if not isinstance(papers, list):
                    print("Error: JSON file must contain a list of objects.")
                    return
            except json.JSONDecodeError:
                print("Error: Invalid JSON format.")
                return

        modified = False  # Track if any change occurs

        for paper in papers:
            title = paper.get("title", "").strip()
            abstract = paper.get("abstract", "").strip()

            # Skip papers that already have a label
            if "label" in paper and paper["label"] in ANNOTATION_CATEGORIES:
                print(f"Skipping already classified paper: {title} ({paper['label']})")
                continue

            if not title or not abstract:
                print(f"Skipping paper due to missing title or abstract: {paper}")
                paper["label"] = "Missing Data"
                modified = True
                continue

            print(f"Classifying paper: {title}")
            label = get_annotation_from_gemini(title, abstract)

            if label == "Error":
                print(f"Skipping paper due to classification error: {title}")
                paper["label"] = "Error"
            else:
                paper["label"] = label
                print(f"Assigned label: {label}")

            modified = True
            time.sleep(10)  # Respect rate limits

            # **Write immediately after classification to avoid losing progress**
            with open(json_file_path, "w", encoding="utf-8") as file:
                json.dump(papers, file, indent=4, ensure_ascii=False)

        if modified:
            print(f"Classification completed! Results saved in {json_file_path}")
        else:
            print("No modifications made. File not updated.")

    except Exception as e:
        print(f"Error processing JSON file: {e}")

if __name__ == "__main__":
    input_json_file = "output.json"
    classify_papers_in_json(input_json_file)
