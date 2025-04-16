# app.py
import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
import time

# Set page configuration
st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for API key input
with st.sidebar:
    st.title("🏥 AI Medical Assistant")
    st.markdown("Powered by CrewAI and Groq")
    
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    
    st.markdown("### About")
    st.info(
        "This application uses multiple AI specialists to provide comprehensive "
        "medical assessments. Each specialist has a different role and expertise, "
        "collaborating to deliver detailed medical guidance."
    )
    
    st.markdown("### Disclaimer")
    st.warning(
        "This application is for demonstration purposes only. "
        "Always consult with qualified healthcare professionals for medical advice."
    )

# Main content
st.title("AI Medical Consultation Assistant")

# Create tabs
tab_patient, tab_results = st.tabs(["Patient Information", "Consultation Results"])

with tab_patient:
    st.header("Patient Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        patient_name = st.text_input("Patient Name", "Jane Doe")
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=58)
        patient_gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        chief_complaint = st.text_area("Chief Complaint", "Chest pain and shortness of breath for the past 3 days, worse with exertion")
    
    with col2:
        medical_history = st.text_area("Medical History", "Hypertension, Type 2 Diabetes (10 years), High cholesterol")
        current_medications = st.text_area("Current Medications", "Metformin 1000mg twice daily, Lisinopril 20mg once daily, Atorvastatin 40mg once daily")
        family_history = st.text_area("Family History", "Father had heart attack at age 62, Mother had stroke at age 70")
    
    st.header("Vital Signs")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        blood_pressure = st.text_input("Blood Pressure", "150/90")
    with col2:
        heart_rate = st.text_input("Heart Rate", "88")
    with col3:
        respiratory_rate = st.text_input("Respiratory Rate", "18")
    with col4:
        temperature = st.text_input("Temperature (°F)", "98.6")
    with col5:
        oxygen_saturation = st.text_input("O2 Saturation (%)", "97")
    
    # Specialist selection
    st.header("Select Specialists for Consultation")
    col1, col2 = st.columns(2)
    
    with col1:
        use_gp = st.checkbox("General Practitioner", value=True)
        use_cardiologist = st.checkbox("Cardiologist", value=True)
    
    with col2:
        use_nutritionist = st.checkbox("Nutritionist", value=True)
        use_pharmacist = st.checkbox("Pharmacist", value=True)
    
    start_consultation = st.button("Start Consultation", type="primary")

with tab_results:
    st.header("Consultation Results")
    if 'consultation_results' not in st.session_state:
        st.session_state.consultation_results = {}
        st.info("Complete the patient information and start the consultation to see results.")
    else:
        if st.session_state.consultation_results:
            for specialist, assessment in st.session_state.consultation_results.items():
                with st.expander(f"{specialist} Assessment", expanded=True):
                    st.markdown(assessment)

# Process the consultation when requested
if start_consultation and groq_api_key:
    # Switch to results tab
    tab_results.active = True
    
    with st.spinner("Medical AI specialists are analyzing the case..."):
        try:
            # Set API key
            os.environ["GROQ_API_KEY"] = groq_api_key
            
            # Format patient information
            patient_info = f"""
            Patient: {patient_name}
            Age: {patient_age}
            Gender: {patient_gender}
            Chief Complaint: {chief_complaint}
            Medical History: {medical_history}
            Current Medications: {current_medications}
            Family History: {family_history}
            Vital Signs: BP {blood_pressure}, HR {heart_rate}, RR {respiratory_rate}, Temp {temperature}°F, O2 Sat {oxygen_saturation}% on room air
            """
            
            # Initialize LLM
            llm = ChatGroq(model="llama3-70b-8192")
            
            # Initialize agents
            agents = []
            tasks = []
            
            # Create specialists based on selection
            if use_gp:
                general_practitioner = Agent(
                    role="General Practitioner",
                    goal="Provide accurate initial assessment of patient symptoms and determine if specialty consultation is needed",
                    backstory="""You are an experienced general practitioner with extensive knowledge in various medical fields.
                    Your primary responsibility is to assess patient symptoms, provide initial diagnoses, and determine
                    if the patient needs to be referred to a specialist.""",
                    verbose=True,
                    allow_delegation=True,
                    llm=llm
                )
                agents.append(general_practitioner)
                
                initial_assessment_task = Task(
                    description=f"""
                    Review the patient information and provide an initial assessment:
                    {patient_info}
                    
                    1. What are the potential diagnoses for this patient?
                    2. What additional tests or information would you recommend?
                    3. Which specialists should be consulted?
                    
                    Provide a comprehensive assessment with clear reasoning.
                    """,
                    expected_output="A detailed initial medical assessment with potential diagnoses, recommended tests, and specialist referrals.",
                    agent=general_practitioner
                )
                tasks.append(initial_assessment_task)
            
            context_tasks = tasks.copy()
            
            if use_cardiologist:
                cardiologist = Agent(
                    role="Cardiologist",
                    goal="Provide expert cardiology assessment and recommendations",
                    backstory="""You are a board-certified cardiologist with 15+ years of experience.
                    You specialize in diagnosing and treating heart conditions, including coronary artery disease,
                    heart failure, arrhythmias, and valvular heart disease.""",
                    verbose=True,
                    allow_delegation=False,
                    llm=llm
                )
                agents.append(cardiologist)
                
                cardiology_consultation_task = Task(
                    description=f"""
                    Provide a cardiology consultation for this patient:
                    {patient_info}
                    
                    Based on the available information, evaluate the cardiac symptoms and risk factors.
                    Recommend appropriate cardiac tests and potential treatments.
                    Explain the cardiac implications of the patient's existing conditions.
                    
                    Provide specific recommendations for this patient's cardiac health.
                    """,
                    expected_output="A specialized cardiology assessment with diagnostic impressions, recommended cardiac tests, and treatment plan.",
                    agent=cardiologist,
                    context=context_tasks if context_tasks else []
                )
                tasks.append(cardiology_consultation_task)
                context_tasks.append(cardiology_consultation_task)
            
            if use_nutritionist:
                nutritionist = Agent(
                    role="Nutritionist",
                    goal="Provide evidence-based dietary and nutritional guidance",
                    backstory="""You are a clinical nutritionist with expertise in dietary interventions for various health conditions.
                    You provide personalized nutritional advice based on patient's health status, medical history, and lifestyle.""",
                    verbose=True,
                    allow_delegation=False,
                    llm=llm
                )
                agents.append(nutritionist)
                
                nutrition_guidance_task = Task(
                    description=f"""
                    Provide nutritional guidance for this patient:
                    {patient_info}
                    
                    Consider the patient's health conditions and symptoms.
                    Provide specific dietary recommendations to improve their overall health.
                    Suggest practical meal planning advice considering all their health conditions.
                    
                    Create a balanced nutritional plan that addresses all health concerns.
                    """,
                    expected_output="A comprehensive nutritional plan with dietary recommendations tailored to the patient's medical conditions.",
                    agent=nutritionist,
                    context=context_tasks if context_tasks else []
                )
                tasks.append(nutrition_guidance_task)
                context_tasks.append(nutrition_guidance_task)
            
            if use_pharmacist:
                pharmacist = Agent(
                    role="Pharmacist",
                    goal="Provide medication information, check drug interactions, and explain proper usage",
                    backstory="""You are a clinical pharmacist with deep knowledge of pharmaceuticals, their mechanisms,
                    side effects, and interactions. You ensure patients understand how to take their medications correctly
                    and safely.""",
                    verbose=True,
                    allow_delegation=False,
                    llm=llm
                )
                agents.append(pharmacist)
                
                medication_review_task = Task(
                    description=f"""
                    Review the patient's medications and provide guidance:
                    {patient_info}
                    
                    Analyze potential drug interactions with their current medications.
                    Suggest any adjustments based on the patient's conditions.
                    Provide detailed instructions on medication timing, dosage, and potential side effects.
                    
                    Ensure the medication plan is comprehensive and addresses all health concerns.
                    """,
                    expected_output="A detailed medication review with analysis of drug interactions, dosing recommendations, and side effect management.",
                    agent=pharmacist,
                    context=context_tasks if context_tasks else []
                )
                tasks.append(medication_review_task)
            
            if not agents or not tasks:
                st.error("Please select at least one specialist.")
            else:
                # Create the medical crew
                medical_crew = Crew(
                    agents=agents,
                    tasks=tasks,
                    verbose=2,
                    process=Process.sequential  # Tasks will be executed in the defined order
                )
                
                # Run the consultation
                results = medical_crew.kickoff()
                
                # Process and display results
                st.session_state.consultation_results = {}
                
                # Extract individual specialist assessments from the crew's output
                output_segments = []
                current_segment = {"specialist": "", "content": ""}
                
                results_lines = results.split('\n')
                specialist_names = ["General Practitioner Assessment", "Cardiologist Assessment", 
                                   "Nutritionist Assessment", "Pharmacist Assessment"]
                
                current_specialist = None
                
                for line in results_lines:
                    # Check if this line indicates a new specialist section
                    found_specialist = False
                    for specialist in specialist_names:
                        if specialist in line or f"{specialist.split()[0]} Evaluation" in line or f"{specialist.split()[0]} Consultation" in line:
                            if current_specialist:
                                # Save previous specialist content
                                st.session_state.consultation_results[current_specialist] = current_segment["content"]
                                
                            # Start new specialist section
                            current_specialist = specialist.split()[0]
                            current_segment = {"specialist": current_specialist, "content": line + "\n"}
                            found_specialist = True
                            break
                    
                    if not found_specialist and current_specialist:
                        # Continue adding content to current specialist
                        current_segment["content"] += line + "\n"
                
                # Add the last specialist's content
                if current_specialist:
                    st.session_state.consultation_results[current_specialist] = current_segment["content"]
                
                # If no structured format was detected, show the full output
                if not st.session_state.consultation_results:
                    st.session_state.consultation_results = {"Combined Assessment": results}
                
                st.success("Consultation completed!")
                
                # Force a rerun to update the Results tab
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.consultation_results = {"Error": f"Failed to complete consultation: {str(e)}"}

elif start_consultation and not groq_api_key:
    st.sidebar.error("Please enter your Groq API key to start the consultation.")

# requirements.txt file content
"""
streamlit==1.28.0
crewai==0.28.0
langchain-groq==0.1.0
"""