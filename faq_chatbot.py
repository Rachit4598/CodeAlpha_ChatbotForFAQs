from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
frequently_asked_questions = {
    "What is Artificial Intelligence?":
    "Artificial Intelligence is the simulation of human intelligence by machines.",

    "What is Machine Learning?":
    "Machine Learning is a subset of Artificial Intelligence that enables systems to learn from data.",

    "What is Deep Learning?":
    "Deep Learning is a subset of Machine Learning based on artificial neural networks.",

    "What is Python?":
    "Python is a high-level programming language widely used in software development, AI, and data science.",

    "Who developed Python?":
    "Python was developed by Guido van Rossum and first released in 1991.",

    "What is Data Science?":
    "Data Science is the process of extracting useful insights and knowledge from data.",

    "What is Natural Language Processing?":
    "Natural Language Processing allows computers to understand and process human language.",

    "What is Computer Vision?":
    "Computer Vision enables computers to interpret and analyze images and videos.",

    "What is a Chatbot?":
    "A chatbot is a software application that simulates human conversation.",

    "What is CodeAlpha?":
    "CodeAlpha provides internship opportunities and project-based learning experiences."
}
question_list = list(frequently_asked_questions.keys())

text_vectorizer = TfidfVectorizer()

question_vectors = text_vectorizer.fit_transform(question_list)


print("=" * 60)
print("FAQ CHATBOT USING NLP")
print("=" * 60)
print("Type your question below.")
print("Type 'exit' to close the chatbot.")
print("=" * 60)


while True:

    user_question = input("\nYou: ")

    if user_question.lower() == "exit":
        print("\nBot: Thank you for using the FAQ Chatbot.")
        break

    user_question_vector = text_vectorizer.transform(
        [user_question]
    )

    similarity_scores = cosine_similarity(
        user_question_vector,
        question_vectors
    )

    most_similar_question_index = similarity_scores.argmax()

    highest_similarity_score = similarity_scores[
        0,
        most_similar_question_index
    ]

    if highest_similarity_score < 0.20:

        print(
            "\nBot: Sorry, I could not find a relevant answer."
        )

    else:

        chatbot_response = frequently_asked_questions[
            question_list[
                most_similar_question_index
            ]
        ]

        print("\nBot:", chatbot_response)
