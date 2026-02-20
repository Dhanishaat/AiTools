from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import whisper
import tempfile
import subprocess
import os
import json
import re
import torch
# from transformers import BartTokenizer, BartForConditionalGeneration
# from transformers import T5Tokenizer, T5ForConditionalGeneration
import sentence_transformers
import faiss
import numpy as np
from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


embedder = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")

faiss_index = None
lecture_chunks = []


model = whisper.load_model("base", device="cpu")

qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2"
)

# 🔹 Explanation model (used ONLY after QA extraction)
# explain_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
# explain_model = T5ForConditionalGeneration.from_pretrained(
#     "google/flan-t5-base",
#     torch_dtype=torch.float32
# )
# explain_model.eval()


summary_tokenizer = AutoTokenizer.from_pretrained(
    "yatharth97/T5-base-10K-summarization"
)

summary_model = AutoModelForSeq2SeqLM.from_pretrained(
    "yatharth97/T5-base-10K-summarization"
)

summary_model.eval()


def upload_page(request):
    return render(request, "upload.html")

@csrf_exempt
def transcribe_audio(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required"})

    media = request.FILES.get("media")
    if not media:
        return JsonResponse({"error": "No file uploaded"})

    # Save uploaded file (audio OR video)
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        for chunk in media.chunks():
            temp.write(chunk)
        input_path = temp.name

    audio_path = input_path + ".wav"

    # 🔹 Always extract audio (works for audio + video)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]

    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        result = model.transcribe(audio_path, language="en")
        text = result.get("text", "").strip()

        if not text:
            return JsonResponse({"error": "No speech detected"})

        summary_points = generate_bullet_summary(text)

    # 🔹 BUILD RAG INDEX (MUST BE BEFORE RETURN)
        global faiss_index, lecture_chunks

        lecture_chunks = chunk_text_for_rag(text)
        embeddings = embedder.encode(lecture_chunks,convert_to_numpy=True,normalize_embeddings=True)

        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)  # 🔥 cosine similarity
        faiss_index.add(embeddings)

        # print("DEBUG CHUNKS:", lecture_chunks[:2])

        return JsonResponse({
            "text": text,
            "summary_points": summary_points
        })


    except Exception as e:
        return JsonResponse({
            "error": "Audio processing failed",
            "details": str(e)
        })

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)

def clean_lecture_text(text):
    text = re.sub(r'\b(uh|um|you know|basically|actually|so yeah)\b', '', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()



def chunk_text(text, max_words=250):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

def chunk_text_for_rag(text, chunk_size=160, overlap=40):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

def answer_question_chunked(question, context_chunks, min_score=0.25):
    best_answer = ""
    best_score = 0

    for chunk in context_chunks:
        if not chunk.strip():
            continue

        result = qa_pipeline(
            question=question,
            context=chunk
        )

        if result["score"] > best_score:
            best_score = result["score"]
            best_answer = result["answer"]

    if best_score < min_score or best_answer.strip() == "":
        return "Not covered in this lecture."

    return best_answer


def generate_bullet_summary(text, max_points=12):
    summaries = []

    for chunk in chunk_text(text, max_words=700):
        prompt = (
            "Create clear, factual student notes from the following lecture. "
            "Do not add new information. Keep it concise.\n\n"
            f"{chunk}"
        )

        inputs = summary_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        with torch.no_grad():
            output_ids = summary_model.generate(
                inputs["input_ids"],
                max_length=300,
                min_length=180,
                num_beams=4,
                temperature=0.3,
                repetition_penalty=1.2,
                early_stopping=True
            )

        summary = summary_tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        summaries.append(summary)

    combined = " ".join(summaries)

    # 🔹 Convert summary into bullet points
    sentences = re.split(r'(?<=[.!?])\s+', combined)

    bullets, seen = [], set()
    for s in sentences:
        s = s.strip()
        if 10 <= len(s.split()) <= 25 and s.lower() not in seen:
            bullets.append(f"• {s}")
            seen.add(s.lower())
        if len(bullets) >= max_points:
            break

    return bullets


# def generate_bullet_summary(text, max_points=8):
#     summaries = []

#     # PASS 1 — factual compression
#     for chunk in chunk_text(text, max_words=200):
#         prompt = (
#             "Summarize the following lecture content factually. "
#             "Keep only important ideas. Do not add explanations.\n\n"
#             f"{chunk}"
#         )

#         inputs = bart_tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=True,
#             max_length=256
#         )

#         with torch.no_grad():
#             summary_ids = bart_model.generate(
#                 inputs["input_ids"],
#                 max_length=120,
#                 min_length=50,
#                 num_beams=6,
#                 no_repeat_ngram_size=3,
#                 early_stopping=True
#             )

#         summaries.append(
#             bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         )

#     combined = " ".join(summaries)

#     # PASS 2 — student-note bullet shaping
#     final_prompt = (
#         "Convert the following lecture summary into clear student notes. "
#         "Each point should be factual and concise.\n\n"
#         f"{combined}"
#     )

#     inputs = bart_tokenizer(
#         final_prompt,
#         return_tensors="pt",
#         truncation=True,
#         max_length=512
#     )

#     with torch.no_grad():
#         final_ids = bart_model.generate(
#             inputs["input_ids"],
#             max_length=180,
#             min_length=80,
#             num_beams=6,
#             no_repeat_ngram_size=3
#         )

#     final_text = bart_tokenizer.decode(final_ids[0], skip_special_tokens=True)

#     sentences = re.split(r'(?<=[.!?])\s+', final_text)

#     bullets, seen = [], set()
#     for s in sentences:
#         s = s.strip()
#         if 8 <= len(s.split()) <= 22 and s.lower() not in seen:
#             bullets.append(f"• {s}")
#             seen.add(s.lower())
#         if len(bullets) >= max_points:
#             break

#     return bullets

@csrf_exempt
def summarize_text(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required"})

    data = json.loads(request.body)
    text = data.get("text", "")

    if not text:
        return JsonResponse({"error": "No text provided"})

    text = clean_lecture_text(text)
    summary_points = generate_bullet_summary(text)

    return JsonResponse({"summary_points": summary_points})

def get_relevant_context(text, question, max_sentences=5):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    question_words = set(question.lower().split())

    scored = []
    for sent in sentences:
        score = sum(1 for w in question_words if w in sent.lower())
        if score > 0:
            scored.append((score, sent))

    scored.sort(reverse=True)
    return " ".join(s for _, s in scored[:max_sentences])


# def retrieve_context_rag(question, top_k=5):
#     if faiss_index is None:
#         return ""

#     q_embedding = embedder.encode([question], convert_to_numpy=True)
#     distances, indices = faiss_index.search(q_embedding, top_k)

#     retrieved_chunks = [lecture_chunks[i] for i in indices[0]]
#     return " ".join(retrieved_chunks)

def retrieve_chunks_rag(question, top_k=5):
    if faiss_index is None:
        return []

    q_embedding = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = faiss_index.search(q_embedding, top_k)

    return [lecture_chunks[i] for i in indices[0]]


def split_into_sentences(text, max_sentences=6):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sentences[:max_sentences])


# @csrf_exempt
# def lecture_chatbot(request):
#     if request.method != "POST":
#         return JsonResponse({"error": "POST request required"})

#     try:
#         data = json.loads(request.body)
#         context = data.get("context", "")
#         question = data.get("question", "")

#         if not context or not question:
#             return JsonResponse({"error": "Missing context or question"})

#         # 🔹 IMPORTANT: reduce context
#         relevant_context = retrieve_context_rag(question)

#         prompt = (
#            '''You are a university lecture assistant. 
#             Answer the question strictly using the context below. Use the same terminology as the lecture.
#             If the answer is partially present, explain briefly.
#             If it is truly missing, say: Not covered in this lecture.\n\n'''
#             f"Context:\n{relevant_context}\n\n"
#             f"Question: {question}\n\n"
#             "Answer is strictly in one sentence only please..not more not less and answer should be accurate and sentence should be completed and clear"
#         )



#         inputs = chat_tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=True,
#             max_length=512
#         )

#         with torch.no_grad():
#             outputs = chat_model.generate(
#                 inputs["input_ids"],
#                 max_length=140,
#                 num_beams=4,
#                 temperature=0.7,
#                 do_sample=True
#             )

#         answer = chat_tokenizer.decode(
#             outputs[0],
#             skip_special_tokens=True
#         )

#         return JsonResponse({"answer": answer})

#     except Exception as e:
#         return JsonResponse({
#             "error": "Chatbot failed",
#             "details": str(e)
#         })

@csrf_exempt
def lecture_chatbot(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required"})

    try:
        data = json.loads(request.body)
        question = data.get("question", "")

        if not question:
            return JsonResponse({"error": "Question is required"})

        # 🔹 Retrieve relevant chunks using FAISS
        relevant_chunks = retrieve_chunks_rag(question)

        if not relevant_chunks:
            return JsonResponse({"answer": "No lecture content available."})

        # 🔹 Extract answer (NO GENERATION)
        answer = answer_question_chunked(question, relevant_chunks)

        return JsonResponse({
            "answer": answer
        })

    except Exception as e:
        return JsonResponse({
            "error": "Chatbot failed",
            "details": str(e)
        })
