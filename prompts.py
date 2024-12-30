from langchain.prompts import PromptTemplate

edge_relevance_prompt = PromptTemplate(
    input_variables=["query", "start_protein", "end_protein", "start_annotation", "end_annotation"],
    template="""As a protein interaction expert, evaluate the relevance of the interaction between {start_protein} and {end_protein}
    to the query: "{query}".

    You should follow those steps for analysis:
    <Steps>
    1. Use the name of proteins along the path to figure out the directional graph. Consider the direction of the interaction and always and only consider the impact of the direct last protein.
    The effect propogates backward to the start protein.
    2. The previous protein should be able to directly interact with the current protein according to the information provided. If not, the interaction is not relevant.
    3. Explan potential biological processes or mechanisms before recommending it. If you can not find any explict reasons and just say they are relevant. The path will be deleted.
    4. The proteins on the ends of one edge must be able to directly interact with each other. If not, the edge does not make any sense and the relevance is zero.
    5. The effect propogates from end protein to start protein.
    </Steps>

    Protein information:
    {start_protein}: {start_annotation}
    {end_protein}: {end_annotation}

    Explain why this interaction is relevant or not relevant to the query using less than 35 words.
    Try to explan the edge informatively using simple and short sentences structure to improve the readability.
    """
)

path_explanation_prompt = PromptTemplate(
    input_variables=["query", "path", "path_details"],
    template="""As a protein-protein interaction expert, choose the following protein interaction paths to satisfy the query: "{query}"
    You should follow those steps for analysis:
    <Steps>
    1. Use the name of proteins along the path to figure out the directional graph. Protein A -> Protein B -> Protein C means that protein c can influence protein b and protein b can influence protein a.
    Any other direction is not relevant.
    2. The path should satisfy the requirements in the query. The more likely the requirements get satisfied, the higher the relevance score is.
    3. The previous protein should be able to directly interact with the current protein according to the information provided. If not, the interaction is not relevant.
    4. Explain potential biological processes or mechanisms before recommending it. If you can not find any explict reasons and just say they are relevant. The path will be deleted.
    5. Provide a relevance score from 0 to 100, where 0 is not relevant at all and 100 is highly relevant This score will be used to rank the paths according to their relevance of the explanation to the query.
    6. The proteins on the ends of one edge must be able to directly interact with each other. If not, the path does not make any sense and the relevance is zero.
    7. The effect propogates from end protein to start protein.
    </Steps>

    Path: {path}

    Details of each step:
    {path_details}

    Discuss any potential biological processes or mechanisms that this path might represent and explain why you recommnend this path, using less than 80 words.

    Your explanation should be informative and accessible, as if you're explaining it to a fellow researcher.
    Your response MUST be in the following JSON format:
    {{
        "explanation": "Your explanation here",
        "relevance_score": "0-100"
    }}
    """
)