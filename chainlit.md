# Welcome to a Retrieval Augmented Generation Chatbot for medical device regulation! 🚀🤖

## Introduction 📚
This is a chatbot that will answer questions about information present in the document `Medical Device Regulation (MDR)`.

## Methodology 🧠
This is an abbreviated version of the pipeline used for the chatbot:
- The question posed by the user is first passed through a Language Model (LLM) to generate a question that will be used to search the database.
- The LLM will also help generate the actual query to the database, with the ability to use metadata filters, for example, filtering by page number.
- Similarity search is performed across the document to find the most relevant information in comparison to the given question.
- The model then generates an answer based on the retrieved information.

## How to use 🤔
You should ask questions about the document `Medical Device Regulation (MDR)`. The questions should be posed in English.

### RAG limitations and improvements

Usually, the questions proposed are directly searched on the database for similar words. This means that something like "Describe the first page of the document" will not work. In that case, the posed questions should be specific and related directly to the contents of the document.

#### Improvements 🌟
However, in this case, we perform a first visit to the LLM to generate the question for looking through the database. This means that, for example, a question like "What is in the first page?" can actually work.

#### Actual limitations 🛑
When referencing the first page, the model will filter by page `1`, which is actually the second page of the document. The actual first page is the cover page, which is actually denoted as page `0`.

To keep consistency among the labels of the pages in the actual document and in our source referencing, the labels of the shown pages are labelled as in the document, starting by 1 in the cover page. So, if you do reference the first page, you will actually receive sources from the second page, which is, indeed, the first page with actual content in the document.

## Useful Links 🔗

The model is basing its answers on english version of the pdf document present at https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex%3A32017R0745. More specifically, the PDF comes from: https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32017R0745
