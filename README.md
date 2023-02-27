## Overview of the problem solved

In this project, [streamlit](https://streamlit.io/) was used to apply a program which is able to extract insights and themes from a transcript (three (3) transcripts in all) and list the insights for each transcript. It also displays a collection of themes (collection of related or similar insights) across the three transcripts. 

The program is currently being hosted at https://ganiyuolalekan-insight7-demo-app-7khl5z.streamlit.app/

## Overview of the program

The program is able to: 
1. Pick the insights from each transcript ✅
2. filter out insights based on how relevant they are to the transcript (not all points contain insights). ✅
3. Able to correct grammatical errors using the grammar corrector API. ✅
4. Able to paraphrase sentence so they are comprehensively distinct using the Paraphrasing API. ✅
5. Generates the insights for each transcript. ✅
6. Map insights of different transcript (and theme transcript) to a theme. ✅

## How to run the program

Run the program by checking the "start application checkbox" of sidebar. The output for the program are generate inline with the expectation for the [case](https://docs.google.com/document/d/1S3FnPYewtQAbGWBdLnnCsGGMXhFuIWECqPGcZKuQe0I/edit).

You can find the code for the application and the code for finding the insights in this [repository](https://github.com/ganiyuolalekan/insight7-demo). The transcripts that can be used to test the application can also be found [here](https://github.com/ganiyuolalekan/insight7-demo)

Codes:
- [Application Code](https://github.com/ganiyuolalekan/insight7-demo/blob/main/app.py)
- [Insights & Themes Code](https://github.com/ganiyuolalekan/insight7-demo/blob/main/utils.py)
- [Transcripts](https://github.com/ganiyuolalekan/insight7-demo/tree/main/data)

## References

- [streamlit](https://streamlit.io/)
- [Paraphrasing API](https://rapidapi.com/smodin/api/rewriter-paraphraser-text-changer-multi-language/details)
- [Grammar Correction API](https://rapidapi.com/microsoft-azure-org-microsoft-cognitive-services/api/bing-spell-check2/details)
- [Insight Similarity API](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- 
