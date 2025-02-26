import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import spacy
import re
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import pearsonr
import networkx as nx
from collections import defaultdict
import io
import PyPDF2
import docx
import subprocess
import warnings
import google.generativeai as genai
warnings.filterwarnings('ignore')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def read_file_content(file):
    """
    Attempt to read file content with different encodings and handle errors
    """
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    content = None
    
    for encoding in encodings:
        try:
            file.seek(0)
            content = file.read().decode(encoding)
            return content
        except UnicodeDecodeError:
            continue
    
    if content is None:
        raise ValueError(f"Could not decode file {file.name} with any of the attempted encodings")

class MedicalReportAnalyzer:
    def __init__(self, use_gemini=False):
        self.reports = []
        self.temporal_data = {}
        self.disease_network = nx.Graph()
        self.use_gemini = use_gemini
        self.gemini_model = None

        if use_gemini:
            try:
                # Initialize Gemini AI with API key from Streamlit secrets
                api_key = st.secrets["gemini_api_key"]
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                st.success("✅ Gemini AI initialized successfully")
            except Exception as e:
                st.error(f"❌ Error initializing Gemini AI: {str(e)}")
                st.error("Make sure your API key is correctly set in .streamlit/secrets.toml file")
                self.use_gemini = False

        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            st.error(f"Error loading spaCy model: {str(e)}")
            st.error("Please run: pip install spacy && python -m spacy download en_core_web_sm")
            raise

        # Initialize other attributes
        self.disease_patterns = self.load_disease_patterns()
        self.risk_factors = self.load_risk_factors()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.label_encoder = LabelEncoder()
        
    def load_disease_patterns(self):
        """Load common disease progression patterns"""
        return {
            'cardiovascular': ['hypertension', 'high cholesterol', 'heart disease', 'stroke'],
            'metabolic': ['obesity', 'diabetes', 'thyroid disorder'],
            'respiratory': ['asthma', 'copd', 'pneumonia', 'bronchitis'],
            'autoimmune': ['rheumatoid arthritis', 'lupus', 'multiple sclerosis'],
            'neurological': ['migraine', 'epilepsy', 'parkinsons', 'alzheimers']
        }
    
    def load_risk_factors(self):
        """Load known risk factors for diseases"""
        return {
            'lifestyle': ['smoking', 'alcohol', 'sedentary', 'diet'],
            'vital_signs': ['blood pressure', 'heart rate', 'bmi', 'temperature'],
            'lab_values': ['cholesterol', 'glucose', 'a1c', 'creatinine'],
            'family_history': ['genetic predisposition', 'family history'],
            'environmental': ['pollution', 'stress', 'occupation', 'exposure']
        }

    def process_file(self, file, report_date):
        """Process individual file and extract content"""
        try:
            file_ext = Path(file.name).suffix.lower()
            content = ""
            
            if file_ext == '.pdf':
                try:
                    pdf_bytes = io.BytesIO(file.read())
                    pdf_reader = PyPDF2.PdfReader(pdf_bytes)
                    content = "\n".join(page.extract_text() for page in pdf_reader.pages)
                except Exception as e:
                    raise ValueError(f"Error processing PDF file: {str(e)}")
            
            elif file_ext == '.txt':
                content = read_file_content(file)
            
            elif file_ext in ['.doc', '.docx']:
                try:
                    doc = docx.Document(io.BytesIO(file.read()))
                    content = "\n".join(paragraph.text for paragraph in doc.paragraphs)
                except Exception as e:
                    raise ValueError(f"Error processing Word file: {str(e)}")
            
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            return self.process_report(content, report_date, file.name)
            
        except Exception as e:
            raise ValueError(f"Error processing file {file.name}: {str(e)}")

    def process_report(self, report_content, report_date, filename=""):
        """Process individual report with temporal tracking"""
        # Convert report_date to string if it's a datetime object for consistent handling
        if isinstance(report_date, datetime):
            report_date = report_date.strftime('%Y-%m-%d')
        
        doc = self.nlp(report_content.lower())
        
        report_data = {
            'date': report_date,
            'filename': filename,
            'diseases': self.extract_diseases(doc),
            'medications': self.extract_medications(doc),
            'vitals': self.extract_vitals(doc),
            'risk_factors': self.extract_risk_factors(doc),
            'lab_results': self.extract_lab_results(doc)
        }
        
        # Use Gemini AI for enhanced analysis if available
        if self.use_gemini and self.gemini_model:
            gemini_insights = self.get_gemini_insights(report_content)
            report_data['gemini_insights'] = gemini_insights
        
        self.reports.append(report_data)
        self.update_temporal_analysis(report_data)
        self.update_disease_network(report_data['diseases'])
        
        return report_data

    def get_gemini_insights(self, report_content):
        """Use Gemini AI to extract additional insights from the report"""
        try:
            prompt = f"""
            As a medical AI assistant, analyze the following medical report and provide insights:
            1. Identify any diseases, conditions, or health concerns that may not be explicitly mentioned
            2. Identify potential risk factors based on the patient's profile
            3. Suggest possible connections between symptoms or conditions
            4. Identify potential treatment recommendations or medication adjustments
            5. Format your response as a structured JSON with these categories
            
            Medical Report:
            {report_content}
            
            Respond with a JSON object with these keys: 
            "identified_conditions", "potential_risks", "condition_connections", "treatment_suggestions"
            """
            
            response = self.gemini_model.generate_content(prompt)
            
            # Process the response to extract JSON
            response_text = response.text
            
            # Find JSON content (assuming it's enclosed in ```json and ```)
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find any JSON-like structure
                json_match = re.search(r'(\{[\s\S]*\})', response_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_text
            
            try:
                insights = json.loads(json_str)
                return insights
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text
                return {"raw_insights": response_text}
                
        except Exception as e:
            st.warning(f"Gemini AI analysis error: {str(e)}")
            return {"error": str(e)}

    def extract_diseases(self, doc):
        """Extract diseases using NLP"""
        diseases = []
        for sent in doc.sents:
            for pattern in self.disease_patterns.values():
                for disease in pattern:
                    if disease in sent.text:
                        diseases.append(disease)
        return list(set(diseases))

    def extract_medications(self, doc):
        """Extract medications and their dosages"""
        medications = []
        med_pattern = r'(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|ml|g)'
        matches = re.finditer(med_pattern, doc.text)
        for match in matches:
            medications.append({
                'name': match.group(1),
                'dosage': match.group(2),
                'unit': match.group(3)
            })
        return medications

    def extract_vitals(self, doc):
        """Extract vital signs with values"""
        vitals = {}
        vital_patterns = {
            'blood_pressure': r'bp[:\s]*(\d+/\d+)',
            'heart_rate': r'hr[:\s]*(\d+)',
            'temperature': r'temp[:\s](\d+\.?\d)',
            'bmi': r'bmi[:\s](\d+\.?\d)'
        }
        
        for vital, pattern in vital_patterns.items():
            match = re.search(pattern, doc.text)
            if match:
                vitals[vital] = match.group(1)
        return vitals

    def extract_risk_factors(self, doc):
        """Extract risk factors from report"""
        risk_factors = []
        for category, factors in self.risk_factors.items():
            for factor in factors:
                if factor in doc.text:
                    risk_factors.append({
                        'category': category,
                        'factor': factor
                    })
        return risk_factors

    def extract_lab_results(self, doc):
        """Extract laboratory results"""
        lab_results = {}
        lab_pattern = r'(\w+)[:\s](\d+(?:\.\d+)?)\s(mg/dl|g/dl|%|u/l)'
        matches = re.finditer(lab_pattern, doc.text)
        for match in matches:
            lab_results[match.group(1)] = {
                'value': float(match.group(2)),
                'unit': match.group(3)
            }
        return lab_results

    def update_temporal_analysis(self, report_data):
        """Update temporal analysis with new report data"""
        date = report_data['date']
        
        if date not in self.temporal_data:
            self.temporal_data[date] = {
                'diseases': set(),
                'risk_factors': set(),
                'lab_trends': {},
                'medication_changes': [],
                'gemini_insights': []
            }
        
        self.temporal_data[date]['diseases'].update(report_data['diseases'])
        
        for risk in report_data['risk_factors']:
            self.temporal_data[date]['risk_factors'].add(
                f"{risk['category']}:{risk['factor']}"
            )
        
        for lab, data in report_data['lab_results'].items():
            if lab not in self.temporal_data[date]['lab_trends']:
                self.temporal_data[date]['lab_trends'][lab] = []
            self.temporal_data[date]['lab_trends'][lab].append(data['value'])
        
        # Add Gemini insights if available
        if 'gemini_insights' in report_data:
            self.temporal_data[date]['gemini_insights'].append(report_data['gemini_insights'])

    def update_disease_network(self, diseases):
        """Update disease correlation network"""
        for disease1 in diseases:
            if not self.disease_network.has_node(disease1):
                self.disease_network.add_node(disease1)
            
            for disease2 in diseases:
                if disease1 != disease2:
                    if self.disease_network.has_edge(disease1, disease2):
                        self.disease_network[disease1][disease2]['weight'] += 1
                    else:
                        self.disease_network.add_edge(disease1, disease2, weight=1)

    def predict_future_diseases(self):
        """Predict potential future diseases based on patterns"""
        if not self.reports:
            return []
        
        current_diseases = set()
        for date in sorted(self.temporal_data.keys()):
            current_diseases.update(self.temporal_data[date]['diseases'])
        
        risk_scores = defaultdict(float)
        
        # Disease network analysis
        for disease in current_diseases:
            if disease in self.disease_network:
                neighbors = self.disease_network[disease]
                for neighbor, data in neighbors.items():
                    if neighbor not in current_diseases:
                        risk_scores[neighbor] += data['weight']
        
        # Analyze lab trends
        lab_trends = self.analyze_lab_trends()
        for lab, trend in lab_trends.items():
            if trend['trend'] == 'increasing' and trend['significance'] < 0.05:
                for disease_category, diseases in self.disease_patterns.items():
                    for disease in diseases:
                        if disease not in current_diseases:
                            risk_scores[disease] += 0.5
        
        # Include Gemini insights for prediction enhancement if available
        if self.use_gemini:
            for date in self.temporal_data:
                for insight_data in self.temporal_data[date].get('gemini_insights', []):
                    if isinstance(insight_data, dict):
                        # Process potential_risks from Gemini
                        potential_risks = insight_data.get('potential_risks', [])
                        if isinstance(potential_risks, list):
                            for risk in potential_risks:
                                for disease_category, diseases in self.disease_patterns.items():
                                    for disease in diseases:
                                        if disease in str(risk).lower() and disease not in current_diseases:
                                            risk_scores[disease] += 0.75
                        
                        # Process identified_conditions from Gemini
                        conditions = insight_data.get('identified_conditions', [])
                        if isinstance(conditions, list):
                            for condition in conditions:
                                for disease_category, diseases in self.disease_patterns.items():
                                    for disease in diseases:
                                        if disease in str(condition).lower() and disease not in current_diseases:
                                            risk_scores[disease] += 0.5
        
        predictions = []
        for disease, score in sorted(risk_scores.items(), key=lambda x: x[1], reverse=True):
            if score > 0.5:  # Lower threshold to include more predictions
                predictions.append({
                    'disease': disease,
                    'risk_score': score,
                    'contributing_factors': self.get_contributing_factors(disease)
                })
        
        return predictions

    def analyze_lab_trends(self):
        """Analyze trends in laboratory values"""
        lab_trends = {}
        
        for date in self.temporal_data:
            for lab, values in self.temporal_data[date]['lab_trends'].items():
                if lab not in lab_trends:
                    lab_trends[lab] = {
                        'values': [],
                        'dates': []
                    }
                lab_trends[lab]['values'].extend(values)
                lab_trends[lab]['dates'].extend([date] * len(values))
        
        results = {}
        for lab, data in lab_trends.items():
            if len(data['values']) > 2:
                try:
                    correlation, p_value = pearsonr(
                        range(len(data['values'])),
                        data['values']
                    )
                    results[lab] = {
                        'trend': 'increasing' if correlation > 0 else 'decreasing',
                        'correlation': correlation,
                        'significance': p_value
                    }
                except Exception as e:
                    st.warning(f"Could not analyze trend for {lab}: {str(e)}")
        
        return results

    def get_contributing_factors(self, disease):
        """Identify factors contributing to disease risk"""
        factors = []
        
        current_risk_factors = set()
        for date in self.temporal_data:
            current_risk_factors.update(self.temporal_data[date]['risk_factors'])
        
        for risk_factor in current_risk_factors:
            category, factor = risk_factor.split(':')
            factors.append({
                'type': 'risk_factor',
                'category': category,
                'factor': factor
            })
        
        lab_trends = self.analyze_lab_trends()
        for lab, trend in lab_trends.items():
            if trend.get('significance', 1.0) < 0.05:
                factors.append({
                    'type': 'lab_trend',
                    'lab': lab,
                    'trend': trend['trend']
                })
        
        # Include Gemini insights as contributing factors if available
        if self.use_gemini:
            for date in self.temporal_data:
                for insight_data in self.temporal_data[date].get('gemini_insights', []):
                    if isinstance(insight_data, dict):
                        connections = insight_data.get('condition_connections', [])
                        if isinstance(connections, list):
                            for connection in connections:
                                if disease.lower() in str(connection).lower():
                                    factors.append({
                                        'type': 'gemini_insight',
                                        'insight_type': 'condition_connection',
                                        'detail': str(connection)
                                    })
        
        return factors

    def generate_visualization(self):
        """Generate visualizations for analysis"""
        figures = []
        
        # Disease progression timeline
        timeline_data = []
        for date in sorted(self.temporal_data.keys()):
            for disease in self.temporal_data[date]['diseases']:
                timeline_data.append({
                    'date': date,
                    'disease': disease
                })
        
        if timeline_data:
            df = pd.DataFrame(timeline_data)
            try:
                fig = px.timeline(
                    df,
                    x_start='date',
                    x_end='date',
                    y='disease',
                    title='Disease Progression Timeline'
                )
                figures.append(fig)
            except Exception as e:
                st.warning(f"Could not generate timeline visualization: {str(e)}")
        
        # Risk factor distribution
        risk_counts = defaultdict(int)
        for date in self.temporal_data:
            for risk in self.temporal_data[date]['risk_factors']:
                risk_counts[risk] += 1
        
        if risk_counts:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(risk_counts.keys()),
                    y=list(risk_counts.values()),
                    name='Risk Factors'
                )
            ])
            fig.update_layout(
                title='Risk Factor Distribution',
                xaxis_title='Risk Factors',
                yaxis_title='Frequency'
            )
            figures.append(fig)
        
        # Disease network visualization using Plotly
        if len(self.disease_network.nodes) > 0:
            try:
                # Calculate node positions using networkx
                pos = nx.spring_layout(self.disease_network, seed=42)
                
                # Create edges trace
                edge_x = []
                edge_y = []
                edge_weights = []
                
                for edge in self.disease_network.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_weights.append(edge[2]['weight'])
                
                edges_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    mode='lines')
                
                # Create nodes trace
                node_x = []
                node_y = []
                node_text = []
                node_size = []
                
                for node in self.disease_network.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                    # Node size based on degree centrality
                    node_size.append(10 + 5 * self.disease_network.degree(node))
                
                nodes_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    text=node_text,
                    marker=dict(
                        showscale=True,
                        colorscale='YlGnBu',
                        size=node_size,
                        colorbar=dict(
                            thickness=15,
                            title='Node Connections',
                            xanchor='left',
                            titleside='right'
                        ),
                        line_width=2))
                
                # Create figure
                fig = go.Figure(data=[edges_trace, nodes_trace],
                             layout=go.Layout(
                                title='Disease Relationship Network',
                                titlefont_size=16,
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                )
                figures.append(fig)
            except Exception as e:
                st.warning(f"Could not generate disease network visualization: {str(e)}")
        
        return figures

    def generate_gemini_summary(self):
        """Generate a comprehensive summary using Gemini AI"""
        if not self.use_gemini or not self.gemini_model:
            return "Gemini AI is not enabled or initialized."
        
        try:
            # Prepare the data for Gemini
            diseases = set()
            risk_factors = set()
            medications = []
            all_insights = []
            
            for report in self.reports:
                diseases.update(report['diseases'])
                medications.extend(report['medications'])
                for risk in report['risk_factors']:
                    risk_factors.add(f"{risk['category']}: {risk['factor']}")
                
                if 'gemini_insights' in report:
                    all_insights.append(report['gemini_insights'])
            
            # Prepare prediction data
            predictions = self.predict_future_diseases()
            pred_text = []
            for pred in predictions:
                factors = []
                for factor in pred['contributing_factors']:
                    if factor['type'] == 'risk_factor':
                        factors.append(f"{factor['category']}: {factor['factor']}")
                    elif factor['type'] == 'lab_trend':
                        factors.append(f"Lab {factor['lab']}: {factor['trend']} trend")
                    elif factor['type'] == 'gemini_insight':
                        factors.append(f"AI insight: {factor['detail']}")
                
                pred_text.append(f"Disease: {pred['disease']}, Risk Score: {pred['risk_score']:.2f}, Factors: {', '.join(factors)}")
            
            prompt = f"""
            As a medical AI assistant, generate a comprehensive summary of the patient's health based on the following data:
            
            Diseases: {', '.join(diseases)}
            Risk Factors: {', '.join(risk_factors)}
            Medications: {[f"{m['name']} {m['dosage']}{m['unit']}" for m in medications]}
            Predictions: {'; '.join(pred_text)}
            
            Additional AI Insights: {json.dumps(all_insights)}
            
            Please provide:
            1. A concise executive summary of the patient's overall health
            2. Key health concerns and their possible interactions
            3. Suggested preventive measures based on the risk factors
            4. Potential treatment options to discuss with healthcare providers
            5. Long-term monitoring recommendations
            
            Format your response in Markdown with clear sections and bullet points where appropriate.
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error generating Gemini summary: {str(e)}"


def create_sample_data():
    """Create sample medical data for demo purposes"""
    return {
        "report1.txt": """
        Patient: John Doe
        Date: 2023-05-15
        
        Medical History: Patient has a history of hypertension and high cholesterol.
        
        Vitals:
        BP: 145/92
        HR: 78
        Temp: 98.6
        BMI: 28.5
        
        Medications:
        Lisinopril 10mg daily
        Simvastatin 20mg nightly
        
        Lab Results:
        Cholesterol: 220 mg/dl
        Glucose: 105 mg/dl
        A1C: 5.8 %
        
        Assessment: Patient continues to struggle with blood pressure control. Recommend lifestyle modifications including reduced sodium intake and regular exercise. Consider increasing lisinopril if BP remains elevated at next visit.
        """,
        
        "report2.txt": """
        Patient: John Doe
        Date: 2023-09-22
        
        Follow-up visit for hypertension and hyperlipidemia.
        
        Patient reports increased stress at work. Has been trying to follow reduced sodium diet but admits to poor compliance.
        
        Vitals:
        BP: 150/95
        HR: 82
        Temp: 98.8
        BMI: 29.1
        
        Medications:
        Lisinopril 20mg daily (increased at last visit)
        Simvastatin 20mg nightly
        
        Lab Results:
        Cholesterol: 210 mg/dl
        Glucose: 118 mg/dl
        A1C: 6.1 %
        Creatinine: 1.1 mg/dl
        
        Assessment: Hypertension remains uncontrolled despite medication increase. Rising glucose and A1C suggest prediabetes. Patient shows risk factors for developing diabetes and needs to focus on weight management.
        
        Plan: Refer to nutritionist, stress management resources, and diabetes prevention program. Schedule follow-up in 3 months with repeat labs.
        """
    }

def main():
    st.set_page_config(page_title="Predictive Medical Analyzer with Gemini AI", layout="wide")
    
    st.title("🏥 Predictive Medical Report Analyzer")
    st.markdown("Upload medical reports for AI-powered predictive analysis")
    
    # Add option to enable Gemini AI
    with st.sidebar:
        st.header("⚙️ Settings")
        use_gemini = st.toggle("Enable Gemini AI", value=True, help="Use Google's Gemini AI for enhanced analysis")
        
        st.header("ℹ About")
        st.markdown("""
        This tool analyzes medical reports to:
        - Track disease progression
        - Identify risk factors
        - Predict potential future health risks
        - Analyze lab result trends
        
        **Now with Gemini AI integration** for advanced medical insights and predictions.
        
        *Supported File Types:*
        - Text files (.txt)
        - PDF documents (.pdf)
        - Word documents (.doc, .docx)
        
        *Note:* All analysis is for informational purposes only and should be verified by healthcare professionals.
        """)
        
        # Add reset button
        if st.button("Reset Analyzer"):
            if 'analyzer' in st.session_state:
                del st.session_state.analyzer
                st.success("Analyzer reset successfully!")
                st.experimental_rerun()
    
    # Initialize the analyzer with Gemini option
    if 'analyzer' not in st.session_state:
        try:
            st.session_state.analyzer = MedicalReportAnalyzer(use_gemini=use_gemini)
        except Exception as e:
            st.error(f"Error initializing analyzer: {str(e)}")
            st.error("Make sure all dependencies are installed correctly.")
            st.info("To install dependencies, run: `pip install streamlit pandas numpy scikit-learn spacy plotly scipy networkx PyPDF2 python-docx google-generativeai`")
            return
    
    # Create tabs for different functions
    tab1, tab2, tab3 = st.tabs(["Upload Reports", "Analysis & Predictions", "Demo"])
    
    with tab1:
        st.header("Upload Medical Reports")
        uploaded_files = st.file_uploader(
            "Upload Medical Reports",
            type=['txt', 'pdf', 'doc', 'docx'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            report_dates = {}
            col1, col2 = st.columns(2)
            
            with col1:
                for i, file in enumerate(uploaded_files):
                    report_dates[file.name] = st.date_input(
                        f"Date for {file.name}",
                        datetime.now()
                    )
            
            with col2:
                st.write("Files to process:")
                for file in uploaded_files:
                    st.write(f"- {file.name}")
            
            if st.button("Process Reports"):
                progress_bar = st.progress(0)
                
                for i, file in enumerate(uploaded_files):
                    try:
                        # Process file with proper error handling
                        st.session_state.analyzer.process_file(file, report_dates[file.name])
                        st.success(f"Successfully processed {file.name}")
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                        continue
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.success("All files processed! Go to the Analysis tab to see results.")
    
    with tab2:
        st.header("Medical Analysis & Predictions")
        
        if not st.session_state.analyzer.reports:
            st.info("No reports have been processed yet. Please upload and process reports first or try the Demo tab.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Predicted Future Health Risks")
                with st.spinner("Generating predictions..."):
                    predictions = st.session_state.analyzer.predict_future_diseases()
                    
                    if predictions:
                      for pred in predictions:
                            exp = st.expander(f"🔍 {pred['disease'].title()} (Risk Score: {pred['risk_score']:.2f})")
                            with exp:
                            st.markdown("**Contributing Factors:**")
                            for factor in pred['contributing_factors']:
                                if factor['type'] == 'risk_factor':
                                    st.markdown(f"- Risk Factor: {factor['category']} - {factor['factor']}")
                                elif factor['type'] == 'lab_trend':
                                    st.markdown(f"- Lab Trend: {factor['lab']} showing {factor['trend']} pattern")
                                elif factor['type'] == 'gemini_insight':
                                    st.markdown(f"- Gemini AI Insight: {factor['detail']}")
                else:
                    st.info("No significant disease predictions found with the current data.")