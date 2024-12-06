import streamlit as st
import pypdf
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    T5TokenizerFast
)
import numpy as np
import json
import time


class PDFQuizSystem:
    def __init__(self):
        # Initialize models with more advanced options
        self.qg_model_name = "google/flan-t5-large"  # More advanced than base T5
        self.mc_model_name = "google/flan-t5-large"  # Using same model for consistency

        # Load question generation model
        self.qg_tokenizer = T5TokenizerFast.from_pretrained(self.qg_model_name)
        self.qg_model = T5ForConditionalGeneration.from_pretrained(
            self.qg_model_name,
            device_map='auto',  # Automatically handle device placement
            torch_dtype=torch.float16  # Use half precision for memory efficiency
        )

        # Load multiple choice generation model
        self.mc_tokenizer = T5TokenizerFast.from_pretrained(self.mc_model_name)
        self.mc_model = T5ForConditionalGeneration.from_pretrained(
            self.mc_model_name,
            device_map='auto',
            torch_dtype=torch.float16
        )

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Performance settings
        self.max_length = 512  # Maximum sequence length
        self.batch_size = 4  # Batch processing for better performance

    @st.cache_data
    def extract_text_from_pdf(self, pdf_file):
        """Extract text content from uploaded PDF file with caching"""
        pdf_content = ""
        pdf_reader = pypdf.PdfReader(pdf_file)

        for page in pdf_reader.pages:
            pdf_content += page.extract_text() + "\n"

        return pdf_content

    def preprocess_text(self, text):
        """Preprocess text into manageable chunks"""
        # Split into sentences (basic implementation)
        sentences = text.replace('\n', ' ').split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        # Combine sentences into chunks of appropriate length
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.max_length:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk)
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def generate_question(self, context):
        """Generate a question from given context using T5"""
        input_text = f"generate question: {context}"

        inputs = self.qg_tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.qg_model.generate(
                inputs.input_ids,
                max_length=64,
                num_beams=4,
                length_penalty=1.5,
                no_repeat_ngram_size=2
            )

        question = self.qg_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return question

    def generate_distractors(self, context, correct_answer):
        """Generate distractors using T5"""
        input_text = f"generate wrong answers for question about: {context} correct answer: {correct_answer}"

        inputs = self.mc_tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.mc_model.generate(
                inputs.input_ids,
                max_length=128,
                num_beams=5,
                num_return_sequences=3,  # Generate 3 distractors
                length_penalty=1.5,
                no_repeat_ngram_size=2
            )

        distractors = [
            self.mc_tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        return list(set(distractors))  # Remove any duplicates

    def generate_mcq(self, context, num_questions=5):
        """Generate multiple choice questions with progress tracking"""
        questions = []
        chunks = self.preprocess_text(context)

        # Create progress bar
        progress_bar = st.progress(0)

        for i, chunk in enumerate(chunks):
            if len(questions) >= num_questions:
                break

            try:
                # Generate question
                question_text = self.generate_question(chunk)

                # Extract answer (using the first sentence of the chunk as correct answer for simplicity)
                correct_answer = chunk.split('.')[0]

                # Generate distractors
                distractors = self.generate_distractors(chunk, correct_answer)

                # Combine options
                options = [correct_answer] + distractors
                np.random.shuffle(options)

                questions.append({
                    "question": question_text,
                    "options": options,
                    "correct_answer": correct_answer
                })

                # Update progress
                progress_bar.progress((i + 1) / min(len(chunks), num_questions))

            except Exception as e:
                st.warning(f"Skipping question due to error: {str(e)}")
                continue

        progress_bar.empty()
        return questions

    def conduct_quiz(self, questions):
        """Present quiz questions and collect responses"""
        responses = []
        score = 0

        for i, q in enumerate(questions):
            st.write(f"\nQuestion {i + 1}: {q['question']}")

            # Create radio buttons for options
            response = st.radio(
                "Select your answer:",
                q['options'],
                key=f"q_{i}"
            )

            responses.append({
                "question": q['question'],
                "selected_answer": response,
                "correct_answer": q['correct_answer']
            })

            if response == q['correct_answer']:
                score += 1

        return responses, score


def main():
    st.title("Advanced PDF Quiz Generator")

    # Initialize session state
    if 'quiz_system' not in st.session_state:
        with st.spinner("Loading AI models... This may take a moment."):
            st.session_state.quiz_system = PDFQuizSystem()
    if 'questions' not in st.session_state:
        st.session_state.questions = None
    if 'quiz_completed' not in st.session_state:
        st.session_state.quiz_completed = False

    # System info
    with st.expander("System Information"):
        st.write(f"Running on: {st.session_state.quiz_system.device}")
        st.write(f"Model: {st.session_state.quiz_system.qg_model_name}")
        st.write("Using FP16 (half precision) for memory efficiency")

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Extract text from PDF
        with st.spinner("Extracting text from PDF..."):
            text_content = st.session_state.quiz_system.extract_text_from_pdf(uploaded_file)

        # Show extracted text
        with st.expander("Show extracted text"):
            st.write(text_content)

        # Quiz generation options
        num_questions = st.slider("Number of questions", 1, 10, 5)

        # Generate questions button
        if st.button("Generate Quiz"):
            start_time = time.time()
            with st.spinner("Generating questions... This may take a few minutes."):
                st.session_state.questions = st.session_state.quiz_system.generate_mcq(
                    text_content,
                    num_questions=num_questions
                )
                end_time = time.time()
                st.success(f"Quiz generated successfully in {end_time - start_time:.2f} seconds!")

        # Conduct quiz if questions are generated
        if st.session_state.questions and not st.session_state.quiz_completed:
            if st.button("Start Quiz"):
                responses, score = st.session_state.quiz_system.conduct_quiz(st.session_state.questions)

                # Display results
                st.write("\n### Quiz Results")
                st.write(f"Score: {score}/{len(st.session_state.questions)}")

                # Save results
                results = {
                    "score": score,
                    "total_questions": len(st.session_state.questions),
                    "responses": responses,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                # Option to download results
                st.download_button(
                    "Download Results",
                    data=json.dumps(results, indent=2),
                    file_name="quiz_results.json",
                    mime="application/json"
                )

                st.session_state.quiz_completed = True

        # Reset button
        if st.session_state.quiz_completed:
            if st.button("Take Another Quiz"):
                st.session_state.questions = None
                st.session_state.quiz_completed = False
                st.experimental_rerun()


if __name__ == "__main__":
    main()