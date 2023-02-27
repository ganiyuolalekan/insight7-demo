from _secrets import *

import re
import time
import requests

import streamlit as st

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import SentenceTransformer, util


def app_meta():
    """Adds app meta data to web applications"""

    # Set website details
    st.set_page_config(
        page_title="Insight7 | Ganiyu Olalekan Result",
        page_icon="images/insight.png",
        layout='centered'
    )

def divider():
    """Sub-routine to create a divider for webpage contents"""

    st.markdown("""---""")


class ProcessTranscript:
    def __init__(
            self, path_to_transcript=None, transcript_data=None, name=None, insight_delimiter='- ',
            relevance_thresh=.3, paraphrase_insights=False, use_grammar_corrector=False
    ):
        """
        Reads the path to the transcript and load it's content
        Checks if a document contains lower than a thousand words
        Extracts theme and insights from the transcript
        Applies grammar correction and/or paraphrasing if permitted.

        NOTE: Using grammar corrections is advised if paraphrasing
        will be set to True

        :param path_to_transcript: str, path to the transcript
        :param transcript_data: str, direct data of the transcript
        :param name: name of the transcript
        :param insight_delimiter: delimiter that identifies an insight
        :param relevance_thresh: threshold to selecting insight relevance
        :param paraphrase_insights: API for paraphrasing insights and themes
        :param use_grammar_corrector: API for correcting grammar in insights and themes
        """

        # Term frequency - inverse document frequency.
        self.tf_idf_model = TfidfVectorizer()

        # Similarity Model
        self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.data = transcript_data
        self.relevance_thresh = relevance_thresh
        self.path_to_transcript = path_to_transcript
        self.insight_delimiter = insight_delimiter
        self.paraphrase_insights = paraphrase_insights
        self.use_grammar_corrector = use_grammar_corrector

        if transcript_data is None and path_to_transcript is not None:
            self.themes, self.insights = self.load_doc()
        elif transcript_data is not None and path_to_transcript is None:
            self.themes, self.insights = self.load_doc(False)
        else:
            raise ValueError("Path and data can't be None")

        self.grader = self.score_grader(self.insights)
        self.insights = [
            insight for insight in self.insights
            if self.grader(self.score_sentence(insight)) >= self.relevance_thresh
        ]
        self.joint_insights = self.insights.copy()

        self.name = name
        self.transcript_collections = {self.name: self.insights}

    @staticmethod
    def correct_grammar(grammar):
        """
        API for correcting grammar

        :param grammar: Sentence to be corrected if there's a grammatical error
        :return: corrected sentence
        """

        response = requests.request(
            "GET", "https://bing-spell-check2.p.rapidapi.com/spellcheck",
            params={
                "mode": 'proof',
                "text": grammar
            },
            headers={
                "X-RapidAPI-Key": RAPID_KEY,
                "X-RapidAPI-Host": RAPID_GRAMMAR_CORRECTOR_HOST
            }
        )

        time.sleep(1)

        res = eval(response.text)

        if len(res['flaggedTokens']):
            for e, s in zip(
                    [e['token'] for e in res['flaggedTokens']],
                    [s['suggestions'][0]['suggestion'] for s in res['flaggedTokens']]
            ):
                grammar.replace(e, s)

        return grammar

    @staticmethod
    def paraphase_grammar(grammar):
        """
        API for para-phasing grammar

        :param grammar: Sentence to be para-phased
        :return: Para-phased grammar
        """

        response = requests.request(
            "POST", "https://rewriter-paraphraser-text-changer-multi-language.p.rapidapi.com/rewrite",
            json={
                "language": "en",
                "strength": 3,
                "text": grammar
            },
            headers={
                "content-type": "application/json",
                "X-RapidAPI-Key": RAPID_KEY,
                "X-RapidAPI-Host": RAPID_PARAPHRASE_HOST
            }
        )

        time.sleep(1)

        return eval(response.text)['rewrite']

    @staticmethod
    def preprocess_text(text, short=3):
        """
        Preprocesses text for NLP operations

        :param text: document or sentence to be processed
        :param short: length of characters to be considered a short word
        :return: processed text
        """

        # Removes special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Removes short words from the string
        text = ' '.join([w for w in text.split() if len(w) > short])

        # Performs word tokenization
        words = word_tokenize(text)

        # Remove stopwords
        stopwords_list = stopwords.words('english')
        words = [
            word for word in words
            if word.lower() not in stopwords_list
        ]

        # Performs lemmatization
        lemma = WordNetLemmatizer()
        words = [
            lemma.lemmatize(word)
            for word in words
        ]

        return ' '.join(words)

    def score_grader(self, sentences):
        """generates a grader for insights"""

        self.tf_idf_model.fit([self.preprocess_text('\n'.join(sentences))])

        scores = [
            self.tf_idf_model.transform([self.preprocess_text(sentence)]).sum()
            for sentence in sentences
        ]

        return lambda score: (score - min(scores)) / (max(scores) - min(scores))

    def score_sentence(self, sentence):
        """scores an insight according to it's relevance to the transcript """

        return self.tf_idf_model.transform([self.preprocess_text(sentence)]).sum()

    def load_doc(self, is_path=True):
        """returns the themes and insights from the transcript"""

        def clean_text(text):
            """Removes spaces, numbers and special character before the text"""

            text = re.sub(r'^[\W\d\s]+', '', text)
            text = re.sub(r'[\W\d\s]+$', '', text)

            return text

        def apply_to_outputs(*outputs, func):
            """Applies a function across values in outputs"""

            return tuple(
                map(lambda outs: [
                    func(out) for out in outs
                ], outputs)
            )

        if is_path:
            with open(self.path_to_transcript, 'r') as f:
                self.data = f.read()

        if not len(self.data.split(' ')) < 1001:
            raise ValueError("Transcript should contain not more than a thousand words")

        insights = []
        themes = []
        entities = self.data.split('\n\n')

        for entity in entities:
            points = entity.split('\n')

            if len(points) == 1 and points[0].startswith(self.insight_delimiter):
                # Single standing insight
                insights.append(points[0])
            elif len(points) == 1 and (not points[0].startswith(self.insight_delimiter)):
                # Single standing theme
                themes.append(points[0])
            elif not points[0].startswith(self.insight_delimiter) and all([
                point.startswith(self.insight_delimiter)
                for point in points[1:]
            ]):
                # Considers Insights with a heading theme
                themes.append(points[0])
                for insight in points[1:]:
                    insights.append(insight)
            elif not points[0].startswith(self.insight_delimiter) and (not all([
                point.startswith(self.insight_delimiter)
                for point in points[1:]
            ])):
                # Considers extended Insights and themes within Insights
                themes.append(points[0])
                for insight_prev, insight_next in zip(points[1:-1], points[2:]):
                    if insight_prev.startswith(self.insight_delimiter) and \
                            insight_next.startswith(self.insight_delimiter):
                        insights.append(insight_prev)
                    elif insight_prev.startswith(self.insight_delimiter) and \
                            insight_next.startswith((' ' * 4) + self.insight_delimiter):
                        insights.append(', '.join([insight_prev, clean_text(insight_next)]))
                    elif insight_prev.startswith((' ' * 4) + self.insight_delimiter) and \
                            insight_next.startswith((' ' * 4) + self.insight_delimiter):
                        insights.append(', '.join([insights.pop(-1), clean_text(insight_next)]))
                    elif insight_prev.startswith((' ' * 4) + self.insight_delimiter) and \
                            insight_next.startswith(self.insight_delimiter):
                        insights.append(insight_next)
                    elif not insight_prev.startswith((' ' * 4) + self.insight_delimiter) and \
                            not insight_next.startswith((' ' * 4) + self.insight_delimiter):
                        themes.append(clean_text(insight_prev))
                    elif not insight_prev.startswith((' ' * 4) + self.insight_delimiter) and \
                            insight_next.startswith(self.insight_delimiter):
                        pass
            else:
                print("Format for extracting themes and insights not included... Please include format for")
                print(f"{'*' * 50}\n\n", entity)

        themes, insights = list(map(set, (themes, insights)))
        themes, insights = list(map(list, (themes, insights)))
        themes, insights = apply_to_outputs(themes, insights, func=clean_text)

        if self.use_grammar_corrector:
            themes, insights = apply_to_outputs(themes, insights, func=self.correct_grammar)

        if self.paraphrase_insights:
            themes, insights = apply_to_outputs(themes, insights, func=self.paraphase_grammar)

        return themes, insights

    def compute_similarity(self, text_1, text_2):
        """computes similarity between two texts"""

        embedding_1 = self.similarity_model.encode(text_1, convert_to_tensor=True)
        embedding_2 = self.similarity_model.encode(text_2, convert_to_tensor=True)

        return round(float(util.pytorch_cos_sim(embedding_1, embedding_2)[0][0]), 4)

    def compute_similarities(self, theme):
        """computes similarity for a single theme"""

        return {
            insight: self.compute_similarity(theme, insight)
            for insight in self.joint_insights
        }

    def compute_bulk_similarities(self):
        """computes similarity for all the themes in the transcript"""

        return {
            theme: self.compute_similarities(theme)
            for theme in self.themes
        }

    def get_theme_similar_insights(self, thresh=.8):
        """

        :param thresh:
        :return:
        """

        def grader(score, _scores):
            """Grades a score to reflex it's relevance"""

            return (score - min(_scores)) / (max(_scores) - min(_scores))

        bulk_compute = self.compute_bulk_similarities()
        related_insights = {}
        for theme in bulk_compute.keys():
            scores = list(bulk_compute[theme].values())
            related_insights[theme] = []

            for insight in bulk_compute[theme].keys():
                if grader(bulk_compute[theme][insight], scores) >= thresh:
                    for name in self.transcript_collections.keys():
                        if insight in self.transcript_collections[name]:
                            related_insights[theme].append([insight, name])
                            break

        return related_insights

    def __add__(self, other):
        self.joint_insights += other.insights
        self.themes += other.themes

        self.transcript_collections[other.name] = other.insights

        return self
