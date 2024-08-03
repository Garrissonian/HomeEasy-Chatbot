# Step 1: Import necessary libraries
import pandas as pd
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BertModel, BertTokenizer
import torch
import faiss
import numpy as np
from langchain.prompts import PromptTemplate

# Load sales data
sales_data = pd.read_csv('sales_performance_data.csv')

# Initialize models and tokenizers
gpt2_model_name = "gpt2"
tokenizer_gpt2 = AutoTokenizer.from_pretrained(gpt2_model_name)
model_gpt2 = AutoModelForCausalLM.from_pretrained(gpt2_model_name)

# Add padding token if not already present
if tokenizer_gpt2.pad_token is None:
    tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token

bert_model_name = "bert-base-uncased"
tokenizer_bert = BertTokenizer.from_pretrained(bert_model_name)
model_bert = BertModel.from_pretrained(bert_model_name)

max_length = tokenizer_gpt2.model_max_length

def truncate_text(text):
    tokens = tokenizer_gpt2.encode(text, truncation=True, max_length=max_length)
    return tokenizer_gpt2.decode(tokens, skip_special_tokens=True)

def generate_text(prompt, max_new_tokens=50):
    truncated_prompt = truncate_text(prompt)
    inputs = tokenizer_gpt2(truncated_prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    attention_mask = inputs.get("attention_mask")
    pad_token_id = tokenizer_gpt2.pad_token_id or tokenizer_gpt2.eos_token_id

    # Check for valid input_ids
    if (inputs["input_ids"] >= tokenizer_gpt2.vocab_size).any():
        raise ValueError("Input contains tokens out of the model's vocabulary range")

    input_length = inputs["input_ids"].size(1)

    outputs = model_gpt2.generate(
        inputs["input_ids"],
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        max_new_tokens=max_new_tokens,  # Ensure additional tokens are generated
        do_sample=True
    )
    return tokenizer_gpt2.decode(outputs[0], skip_special_tokens=True)

def preprocess_text(text):
    inputs = tokenizer_bert(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    # Use the mean of the last hidden states for embeddings
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def create_vector_db(data):
    vectors = [preprocess_text(str(row)) for _, row in data.iterrows()]
    vectors = np.array(vectors).astype(np.float32)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

vector_db = create_vector_db(sales_data)

def generate_rag_response(query):
    query_vector = preprocess_text(query)
    query_vector = np.array([query_vector]).astype(np.float32)
    D, I = vector_db.search(query_vector, k=5)
    retrieved_docs = [sales_data.iloc[i].to_dict() for i in I[0]]
    context = ' '.join([str(doc) for doc in retrieved_docs])
    return generate_text(f"Context: {context}\nQuery: {query}\nAnswer:")

def generate_langchain_response(query):
    context = generate_rag_response(query)
    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template="Context: {context}\nQuery: {query}\nAnswer:"
    )
    return generate_text(prompt.format(context=context, query=query))

# Flask API
app = Flask(__name__)

@app.route('/api/rep_performance', methods=['GET'])
def rep_performance():
    try:
        rep_id = request.args.get('rep_id')
        response = generate_langchain_response(f"Performance of rep {rep_id}")
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, debug=True)
