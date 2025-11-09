from crewai import Agent, LLM, Task, Crew
from crewai_tools import RagTool, SerperDevTool, ScrapeWebsiteTool


# RAG knowledge base (internal PDFs and web pages)
rag_tool = RagTool()
rag_tool.add(data_type="file", path="Code_Assurance_Version_FR.pdf")
rag_tool.add(data_type="file", path="code_arabe.pdf")

rag_tool.add(data_type="web_page", url="https://www.ftusanet.org/")
# grp facebook assurance tunisienne
rag_tool.add(
    data_type="web_page", url="https://www.facebook.com/groups/706953679342592"
)
rag_tool.add(data_type="file", path="facebook_posts.txt")


rag_tool.add(
    data_type="web_page",
    url="https://www.ftusanet.org/cadre-institutionnel/code-des-assurances/",
)
rag_tool.add(
    data_type="web_page", url="https://www.ghorbel.tn/rubrique_actualite/faq.php"
)

rag_tool.add(data_type="web_page", url="https://www.gat.com.tn/faq")

rag_tool.add(data_type="web_page", url="https://www.hayett.tn/faq")

rag_tool.add(data_type="web_page", url="https://www.star.com.tn/faqs?page=4")
rag_tool.add(data_type="web_page", url="https://www.cotunace.com.tn/fr/faq/faq")
rag_tool.add(data_type="web_page", url="https://bh-assurance.com/faq-assurances")
rag_tool.add(
    data_type="web_page",
    url="https://www.maghrebia.com.tn/site/fr/assurance-automobile.25.html",
)
rag_tool.add(
    data_type="web_page",
    url="https://www.comar.tn/faq",
)

# Web search tool (for latest news or regulations)
search_tool = SerperDevTool()

# Scraping tool (for deeper content from specific websites)
scrape_tool = ScrapeWebsiteTool(
    website_url="https://www.cga.gov.tn/index.php?id=102&L=1%2A"
)


llm = LLM(model="huggingface/youssefrekik/qwen3-8b-tunisian-insurance", temperature=0.7)


# Agent 1: RAG retrieval
rag_agent = Agent(
    role="Internal RAG Analyst",
    goal="Retrieve and summarize information from PDFs and web pages in the internal knowledge base.",
    backstory="An expert in insurance regulations and internal documentation.",
    tools=[rag_tool],
    llm=llm,
    verbose=True,
)

# Agent 2: Web search & scrape
web_agent = Agent(
    role="Insurance Web Researcher",
    goal="Fetch recent information about insurance in Tunisia from web searches and scraping websites.",
    backstory="An AI agent that collects up-to-date insurance news and regulations.",
    tools=[search_tool, scrape_tool],
    llm=llm,
    verbose=True,
)

# Agent 3: Synthesis agent
synthesis_agent = Agent(
    role="Insurance Chatbot Synthesizer",
    goal=(
        "Combine results from the RAG agent and Web agent to produce a coherent, clear answer. "
        "Respond in Tunisian Arabic dialect, using casual but professional style."
    ),
    backstory="A conversational agent trained to generate user-friendly responses based on multiple sources.",
    llm=llm,
    verbose=True,
    reasoning=True,
)


# --- 4️⃣ DEFINE TASKS ---

task_rag = Task(
    description="Analyze the internal documents and summarize relevant points.",
    expected_output="A summary of the internal insurance code and related documents.",
    agent=rag_agent,
)

task_web = Task(
    description="Collect recent information from web sources and summarize updates.",
    expected_output="A summary of recent news or changes in insurance regulations.",
    agent=web_agent,
)

task_synthesis = Task(
    description="Combine all gathered information into a final answer for the user's question.",
    expected_output="A clear, concise, and accurate response in Tunisian dialect.",
    agent=synthesis_agent,
)

# --- 5️⃣ ASSEMBLE CREW ---

crew = Crew(
    agents=[rag_agent, web_agent, synthesis_agent],
    tasks=[task_rag, task_web, task_synthesis],
    verbose=True,
    planning=True,
)

# --- 6️⃣ EXECUTE ---

question = "En dialecte tunisien, explique-moi les points clés du Code Assurance Version FR et les nouveautés récentes."
result = crew.kickoff(inputs={"question": question})

print("\n💬 Réponse finale du chatbot d’assurance agentic :\n")
print(result)
