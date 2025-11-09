from crewai import Agent, LLM, Task, Crew
from crewai_tools import SerperDevTool, PDFSearchTool, ScrapeWebsiteTool

# --- 1️⃣ TOOLS SETUP ---

# RAG document (internal insurance code)
rag_tool = PDFSearchTool(pdf="Code_Assurance_Version_FR.pdf")

# Search tool for external info (e.g. updates, new regulations)
search_tool = SerperDevTool()

# Scraping tool for reading content from websites
scrape_tool = ScrapeWebsiteTool()

# Example: restrict to a specific trusted site if needed
# scrape_tool = ScrapeWebsiteTool(website_url='https://www.ftusa.org.tn')  # FTUSA = Fédération Tunisienne des Sociétés d’Assurances

# --- 2️⃣ LLM SETUP ---

# Your fine-tuned Tunisian dialect insurance model
llm = LLM(model="huggingface/youssefrekik/qwen3-8b-tunisian-insurance", temperature=0.7)

# --- 3️⃣ AGENTS DEFINITION ---

# 🌍 Agent 1: Web Researcher
web_research_agent = Agent(
    role="Insurance Web Researcher",
    goal="Trouver des informations récentes et pertinentes sur les assurances en Tunisie depuis le web.",
    backstory="Un analyste curieux qui utilise la recherche web et le scraping pour collecter les dernières informations du marché de l’assurance.",
    tools=[search_tool, scrape_tool],
    llm=llm,
    verbose=True,
)

# 📘 Agent 2: RAG Document Analyst
rag_analyst_agent = Agent(
    role="Internal Insurance Analyst",
    goal="Analyser le document interne 'Code Assurance Version FR' pour extraire les points clés pertinents à la question posée.",
    backstory="Spécialiste des lois d’assurance tunisiennes, expert dans l’analyse du Code des Assurances.",
    tools=[rag_tool],
    llm=llm,
    verbose=True,
)

# 🧠 Agent 3: Synthesizer
synthesis_agent = Agent(
    role="Insurance Synthesis Chatbot",
    goal="Combiner les informations internes et externes pour donner une réponse claire, complète et naturelle en dialecte tunisien.",
    backstory="Un chatbot intelligent formé à résumer et expliquer les informations complexes en langage clair et tunisien.",
    llm=llm,
    verbose=True,
    reasoning=True,
)

# --- 4️⃣ TASKS DEFINITION ---

task_web_search = Task(
    description="Chercher les nouveautés et informations récentes sur les assurances en Tunisie à partir du web.",
    expected_output="Un résumé clair des mises à jour récentes de la législation ou des produits d’assurance tunisiens.",
    agent=web_research_agent,
)

task_rag_analysis = Task(
    description="Analyser le document interne 'Code Assurance Version FR' et en extraire les points clés relatifs aux questions des utilisateurs.",
    expected_output="Un résumé synthétique des points essentiels du Code d’Assurance.",
    agent=rag_analyst_agent,
)

task_synthesis = Task(
    description=(
        "Fusionner les résultats du chercheur web et de l’analyste interne pour donner une réponse finale complète "
        "en dialecte tunisien, expliquant les points clés du Code d’Assurance et les nouveautés récentes."
    ),
    expected_output="Une réponse finale claire, précise et rédigée en dialecte tunisien.",
    agent=synthesis_agent,
)

# --- 5️⃣ CREW ASSEMBLY ---

crew = Crew(
    agents=[web_research_agent, rag_analyst_agent, synthesis_agent],
    tasks=[task_web_search, task_rag_analysis, task_synthesis],
    verbose=True,
    memory=True,
    planning=True,  # Allow agents to plan and communicate
)

# --- 6️⃣ EXECUTION ---

question = "Quels sont les points clés du Code Assurance Version FR et les nouveautés récentes ?"
result = crew.kickoff(inputs={"question": question})

print("\n💬 Réponse finale du chatbot d’assurance agentic :\n")
print(result)
