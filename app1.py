import streamlit as st
import pypdf
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    pipeline
)
import numpy as np
import json
import time
import plotly.express as px
import pandas as pd


class PDFQuizSystem:
    def __init__(self):
        # Initialize models
        self.qa_model_name = "deepset/roberta-base-squad2"
        self.classifier_name = "facebook/bart-large-mnli"

        with st.spinner("Loading AI models... Please wait..."):
            # Load tokenizers and models
            self.qa_tokenizer = AutoTokenizer.from_pretrained(self.qa_model_name)
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(self.qa_model_name)
            self.classifier_tokenizer = AutoTokenizer.from_pretrained(self.classifier_name)
            self.classifier_model = AutoModelForSequenceClassification.from_pretrained(self.classifier_name)

            # Initialize question generation pipeline
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.qa_model,
                tokenizer=self.qa_tokenizer
            )

    def extract_text_from_pdf(self, pdf_file):
        """Extract text content from uploaded PDF file"""
        pdf_content = ""
        pdf_reader = pypdf.PdfReader(pdf_file)

        progress_text = st.empty()
        progress_bar = st.progress(0)
        total_pages = len(pdf_reader.pages)

        for i, page in enumerate(pdf_reader.pages):
            pdf_content += page.extract_text() + "\n"
            progress_bar.progress((i + 1) / total_pages)
            progress_text.text(f"Extracting page {i + 1} of {total_pages}")

        progress_bar.empty()
        progress_text.empty()
        return pdf_content

    def generate_mcq(self, context, num_questions=5):
        """Generate multiple choice questions from the text"""
        chunks = [context[i:i + 512] for i in range(0, len(context), 512)]
        questions = []
        attempts = 0
        max_attempts = num_questions * 2  # Allow more attempts to reach desired question count

        progress_text = st.empty()
        progress_bar = st.progress(0)

        while len(questions) < num_questions and attempts < max_attempts:
            try:
                # Update progress
                progress_text.text(f"Generating question {len(questions) + 1} of {num_questions}")
                progress_bar.progress(len(questions) / num_questions)

                # Get a chunk, cycling through if needed
                chunk_index = attempts % len(chunks)
                chunk = chunks[chunk_index]

                # Generate question
                question = {
                    "question": f"What is the main point discussed in: '{chunk[:100]}...'?",
                    "context": chunk
                }

                answer = self.qa_pipeline(question)

                # Only proceed if we get a meaningful answer
                if len(answer['answer'].strip()) > 10:
                    distractors = self.generate_distractors(chunk, answer['answer'])

                    # Only add question if we got valid distractors
                    if len(distractors) == 3:  # We want exactly 3 distractors
                        options = [answer['answer']] + distractors
                        np.random.shuffle(options)

                        questions.append({
                            "question": question["question"],
                            "options": options,
                            "correct_answer": answer['answer']
                        })

            except Exception as e:
                st.error(f"Error generating question: {str(e)}")

            attempts += 1

        # Ensure we have exactly the number of questions requested
        if len(questions) > num_questions:
            questions = questions[:num_questions]

        # Final progress update
        progress_bar.progress(1.0)
        progress_text.text(f"Generated {len(questions)} questions")
        time.sleep(0.5)  # Brief pause to show completion
        progress_bar.empty()
        progress_text.empty()

        if len(questions) < num_questions:
            st.warning(f"Could only generate {len(questions)} quality questions from the provided content.")

        return questions

    def generate_distractors(self, context, correct_answer, num_distractors=3):
        """Generate wrong options for MCQ"""
        sentences = context.split('.')
        distractors = []

        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10
                           and s.strip() != correct_answer]

        for sentence in valid_sentences:
            if len(distractors) >= num_distractors:
                break

            similarity = self.compute_similarity(sentence, correct_answer)
            if 0.3 < similarity < 0.7:
                distractors.append(sentence)

        while len(distractors) < num_distractors:
            distractors.append(f"None of the above options")

        return distractors[:num_distractors]

    def compute_similarity(self, text1, text2):
        """Compute semantic similarity between two texts"""
        inputs = self.classifier_tokenizer(
            text1,
            text2,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.classifier_model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return probs[:, 1].item()

    def analyze_quiz_performance(self, responses, total_time):
        """Analyze quiz performance in detail"""
        total_questions = len(responses)
        correct_answers = sum(1 for r in responses if r['selected_answer'] == r['correct_answer'])

        score_percentage = (correct_answers / total_questions) * 100
        time_per_question = total_time / total_questions

        question_analysis = []
        for i, response in enumerate(responses):
            question_analysis.append({
                'question_number': i + 1,
                'correct': response['selected_answer'] == response['correct_answer'],
                'selected': response['selected_answer'],
                'correct_answer': response['correct_answer']
            })

        return {
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'score_percentage': score_percentage,
            'time_per_question': time_per_question,
            'question_analysis': question_analysis
        }

    def display_results_dashboard(self, analysis, total_time):
        """Display an interactive results dashboard"""
        st.markdown("## 📊 Quiz Performance Dashboard")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Final Score",
                f"{analysis['score_percentage']:.1f}%",
                f"{analysis['correct_answers']}/{analysis['total_questions']} correct"
            )

        with col2:
            st.metric(
                "Time Taken",
                f"{total_time:.1f} sec",
                f"{analysis['time_per_question']:.1f} sec/question"
            )

        with col3:
            performance_label = "Excellent! 🌟" if analysis['score_percentage'] >= 90 else \
                "Great Job! 🎉" if analysis['score_percentage'] >= 80 else \
                    "Good Work! 👍" if analysis['score_percentage'] >= 70 else \
                        "Keep Practicing! 💪"
            st.metric("Performance", performance_label)

        df = pd.DataFrame(analysis['question_analysis'])
        fig = px.bar(
            df,
            x='question_number',
            y='correct',
            title='Question Performance',
            labels={'question_number': 'Question Number', 'correct': 'Correct'},
            color='correct',
            color_discrete_map={True: '#00CC96', False: '#EF553B'}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📝 Detailed Analysis")
        for q in analysis['question_analysis']:
            with st.expander(f"Question {q['question_number']}"):
                if q['correct']:
                    st.markdown("✅ **Correct!**")
                else:
                    st.markdown("❌ **Incorrect**")
                st.markdown(f"Your answer: {q['selected']}")
                if not q['correct']:
                    st.markdown(f"Correct answer: {q['correct_answer']}")

    def conduct_quiz(self, questions):
        """Enhanced quiz conduction with proper flow and visibility"""
        quiz_container = st.container()

        with quiz_container:
            st.markdown("""
            <div style='text-align: center; background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                <h2>📝 Quiz Time!</h2>
                <p>Answer all questions and click Submit to see your results.</p>
            </div>
            """, unsafe_allow_html=True)

            # Initialize responses in session state if not exists
            if 'responses' not in st.session_state:
                st.session_state.responses = []

            # Display all questions at once
            for i, q in enumerate(questions):
                st.markdown(f"### Question {i + 1} of {len(questions)}")

                # Question card with dark text on light background
                st.markdown(f"""
                <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; 
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0;'>
                    <p style='font-size: 18px; color: #000000;'>{q['question']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Radio buttons for options
                option = st.radio(
                    "Select your answer:",
                    q['options'],
                    key=f"q_{i}",
                    help="Choose one option"
                )

                # Store response
                if len(st.session_state.responses) <= i:
                    st.session_state.responses.append({
                        "question": q['question'],
                        "selected_answer": option,
                        "correct_answer": q['correct_answer']
                    })
                else:
                    st.session_state.responses[i] = {
                        "question": q['question'],
                        "selected_answer": option,
                        "correct_answer": q['correct_answer']
                    }

                st.markdown("---")

            # Submit button at the end of all questions
            if st.button("Submit Quiz"):
                score = sum(1 for r in st.session_state.responses
                            if r['selected_answer'] == r['correct_answer'])

                # Calculate total time
                total_time = time.time() - st.session_state.get('quiz_start_time', time.time())

                # Display results
                performance_analysis = self.analyze_quiz_performance(
                    st.session_state.responses,
                    total_time
                )
                self.display_results_dashboard(performance_analysis, total_time)

                # Return results for saving
                return st.session_state.responses, score

            return [], 0


def main():
    st.set_page_config(page_title="Smart PDF Quiz Generator", layout="wide")

    st.markdown("""
        <style>
        .stButton button {
            width: 100%;
            font-size: 18px;
            padding: 10px;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .question-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        .metric-card {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .success { color: #00CC96; }
        .warning { color: #EF553B; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>📚 Smart PDF Quiz Generator</h1>", unsafe_allow_html=True)
    st.markdown("---")

    if 'quiz_system' not in st.session_state:
        st.session_state.quiz_system = PDFQuizSystem()
    if 'questions' not in st.session_state:
        st.session_state.questions = None
    if 'quiz_completed' not in st.session_state:
        st.session_state.quiz_completed = False

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📁 Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            num_questions = st.slider("Number of questions", 5, 15, 10)

    with col2:
        if uploaded_file is not None:
            text_content = st.session_state.quiz_system.extract_text_from_pdf(uploaded_file)

            with st.expander("📄 Show PDF Content"):
                st.write(text_content)

            if st.button("🎯 Generate Quiz"):
                with st.spinner("🤖 AI is generating your quiz..."):
                    start_time = time.time()
                    st.session_state.questions = st.session_state.quiz_system.generate_mcq(
                        text_content,
                        num_questions=num_questions
                    )
                    end_time = time.time()
                    st.success(f"✨ Quiz generated in {end_time - start_time:.1f} seconds!")

            if st.session_state.questions and not st.session_state.quiz_completed:
                if 'quiz_start_time' not in st.session_state:
                    st.session_state.quiz_start_time = time.time()

                responses, score = st.session_state.quiz_system.conduct_quiz(st.session_state.questions)

                if score > 0:  # Only mark as completed if quiz was submitted
                    st.session_state.quiz_completed = True

                    # Save results
                    results = {
                        "score": score,
                        "total_questions": len(st.session_state.questions),
                        "responses": responses,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }

                    # Option to download results
                    st.download_button(
                        "📥 Download Results",
                        data=json.dumps(results, indent=2),
                        file_name="quiz_results.json",
                        mime="application/json"
                    )

            if st.session_state.quiz_completed:
                if st.button("🔄 Take Another Quiz"):
                    st.session_state.questions = None
                    st.session_state.quiz_completed = False
                    st.session_state.responses = []
                    st.experimental_rerun()


if __name__ == "__main__":
    main()