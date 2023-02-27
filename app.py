import streamlit as st
from io import StringIO
from utils import app_meta, divider, ProcessTranscript


app_meta()

# Initializing Side-Bar
with st.sidebar:
    st.write("")
    start_project = st.checkbox(
        label="Start Application",
        help="Starts The Demo Application"
    )
    divider()

if start_project:
    with st.sidebar:
        st.write("Start by imputing three (3) transcript with no more than a thousand words each")
        divider()

        relevance_thresh = st.slider(
            "Insight Relevance", min_value=.0, max_value=1.00, value=.30, step=.001,
            help="The greater the insight relevance, the more strict the insights picked"
        )
        theme_similarity_thresh = st.slider(
            "Theme Similarity Thresh", min_value=.0, max_value=1.00, value=.80, step=.001,
            help="The greater the theme similarity, the more strict the insights picked"
        )
        use_grammar_corrector = st.checkbox(
            "Use Grammar Correction API", value=False,
            help="Corrects grammatical errors in sentences with grammatical issues."
        )
        paraphrase_insights = st.checkbox(
            "Use Grammar Paraphrasing API", value=False,
            help="Paraphrases grammar so each sentence appears unique"
        )

        st.markdown("> **NOTE**: Using grammar corrector and paraphraser can take a while")
        divider()

        transcript_1 = st.file_uploader("Choose your first transcript file")
        transcript_2 = st.file_uploader("Choose your second transcript file")
        transcript_3 = st.file_uploader("Choose your third transcript file")

    transcript_data = {}
    for i, transcript in enumerate((transcript_1, transcript_2, transcript_3), start=1):
        if transcript is not None:
            transcript_data[i] = StringIO(transcript.getvalue().decode("utf-8")).read()

    if len(transcript_data) < 3:
        st.write("Please upload the transcript to begin, you can get the files from "
                 "[Transcripts]([Transcripts](https://drive.google.com/drive/folders/1K3ULlqH0W45z54qp4SgHMgm5I8j6Mmwb?usp=sharing))")
    else:
        transcript_1 = ProcessTranscript(
            transcript_data=transcript_data[1], name='transcript_1',
            relevance_thresh=relevance_thresh, use_grammar_corrector=use_grammar_corrector,
            paraphrase_insights=paraphrase_insights
        )

        transcript_2 = ProcessTranscript(
            transcript_data=transcript_data[2], name='transcript_2',
            relevance_thresh=relevance_thresh, use_grammar_corrector=use_grammar_corrector,
            paraphrase_insights=paraphrase_insights
        )

        transcript_3 = ProcessTranscript(
            transcript_data=transcript_data[3], name='transcript_3',
            relevance_thresh=relevance_thresh, use_grammar_corrector=use_grammar_corrector,
            paraphrase_insights=paraphrase_insights
        )

        st.write("# Overview of the insights")

        tab1, tab2, tab3 = st.tabs([
            "Insights from the first transcript",
            "insights from the second transcript",
            "insights from the third transcript"
        ])

        with tab1:
            st.header(f"We have a total of {len(transcript_1.insights)} insights")
            for i, insight in enumerate(transcript_1.insights, start=1):
                st.write(f'{i}.' + insight)

        with tab2:
            st.header(f"We have a total of {len(transcript_2.insights)} insights")
            for i, insight in enumerate(transcript_2.insights, start=1):
                st.write(f'{i}.' + insight)

        with tab3:
            st.header(f"We have a total of {len(transcript_3.insights)} insights")
            for i, insight in enumerate(transcript_3.insights, start=1):
                st.write(f'{i}.' + insight)

        divider()

        st.write("# Mapping insights to a theme based on similarity")
        st.write("**👈🏾👈🏾👈🏾 You can use the similarity threshold to control how tightly related they should be**")

        st.write("### We have the following records of themes in the transcripts")
        divider()
        st.write(f"#### The first transcript has {len(transcript_1.themes)} themes which are:")
        for i, theme in enumerate(transcript_1.themes, start=1):
            st.write(f'{i}.' + theme)
        divider()
        st.write(f"#### The second transcript has {len(transcript_2.themes)} themes which are:")
        for i, theme in enumerate(transcript_2.themes, start=1):
            st.write(f'{i}.' + theme)
        divider()
        st.write(f"#### The third transcript has {len(transcript_3.themes)} themes which are:")
        for i, theme in enumerate(transcript_3.themes, start=1):
            st.write(f'{i}.' + theme)
        divider()

        st.write("### Here are the insights for each theme")

        transcript_1 += transcript_2
        transcript_1 += transcript_3

        theme_insight_map = transcript_1.get_theme_similar_insights(thresh=theme_similarity_thresh)

        for theme in theme_insight_map:
            st.write(f"##### Theme: \"{theme}\"")
            for i, result in enumerate(theme_insight_map[theme], start=1):
                st.write(f"{i}. {result[0]}. **FROM {result[1]}**")
            divider()
else:
    with open('README.md', 'r') as f:
        demo_report = f.read()
    st.markdown(demo_report)
