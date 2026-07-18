from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

def load_ner_pipeline(model_path="./final_model"):
    """
    Loads the fine-tuned model and initializes the NER pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    # aggregation_strategy="simple" automatically merges sub-tokens 
    ner_pipe = pipeline(
        "token-classification", 
        model=model, 
        tokenizer=tokenizer, 
        aggregation_strategy="simple"
    )
    return ner_pipe

def extract_mountains(text, ner_pipe):
    """
    Processes the text and extracts all identified mountain names.
    """
    # The pipeline returns a list of dictionaries containing the extracted entities
    entities = ner_pipe(text)
    mountains = []
    
    for entity in entities:
        # Since the model was trained on a single class, we filter for the 'MOUNTAIN' group
        if entity['entity_group'] == 'MOUNTAIN':
            mountains.append({
                "name": entity['word'],
                "score": round(entity['score'], 4),
                "start": entity['start'],
                "end": entity['end']
            })
            
    return mountains

if __name__ == "__main__":
    print("Loading model")
    # The path must point to the directory where train.py saved the final output
    ner_pipeline = load_ner_pipeline(model_path="./final_model")
    
    # Test examples (including a negative example with no mountains)
    test_sentences = [
        "I have never been to Mount Everest, but I hiked Hoverla last summer.",
        "The weather in London is quite rainy today, so we stayed at the hotel.",
        "K2 is known as the Savage Mountain due to the extreme difficulty of ascent."
    ]
    
    print("\nModel Testing")
    for sentence in test_sentences:
        print(f"\nText: {sentence}")
        found_mountains = extract_mountains(sentence, ner_pipeline)
        
        if found_mountains:
            print("Extracted mountains:")
            for m in found_mountains:
                print(f" - {m['name']} (Confidence: {m['score']})")
        else:
            print("No mountains detected.")