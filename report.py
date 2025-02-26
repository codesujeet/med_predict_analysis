import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import spacy
import re
from collections import defaultdict
import io
import json
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Load spaCy model with proper error handling
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

class MedicalReportAnalyzer:
    def __init__(self):
        self.reports = []
        self.temporal_data = {}
        self.disease_network = nx.Graph()
        
        # Initialize NLP
        try:
            self.nlp = load_spacy_model()
        except Exception as e:
            st.error(f"Error loading spaCy model: {str(e)}")
            st.error("Please run: pip install spacy && python -m spacy download en_core_web_sm")
            raise

        # Initialize other attributes
        self.disease_patterns = self.load_disease_patterns()
        self.risk_factors = self.load_risk_factors()
        
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
        """Process text file and extract content"""
        try:
            content = file.getvalue().decode('utf-8')
            return self.process_report(content, report_date, file.name)
        except UnicodeDecodeError:
            try:
                content = file.getvalue().decode('latin-1')
                return self.process_report(content, report_date, file.name)
            except Exception as e:
                raise ValueError(f"Error processing file {file.name}: {str(e)}")

    def process_report(self, report_content, report_date, filename=""):
        """Process individual report with temporal tracking"""
        # Convert report_date to string if it's a datetime object
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
        
        self.reports.append(report_data)
        self.update_temporal_analysis(report_data)
        self.update_disease_network(report_data['diseases'])
        
        return report_data

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
            'temperature': r'temp[:\s](\d+\.?\d*)',
            'bmi': r'bmi[:\s](\d+\.?\d*)'
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
                'medication_changes': []
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
        
        predictions = []
        for disease, score in sorted(risk_scores.items(), key=lambda x: x[1], reverse=True):
            if score > 0.5:  # Threshold for predictions
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
                except Exception:
                    pass # Skip if correlation can't be calculated
        
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
            except Exception:
                pass # Skip if timeline can't be generated
        
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
        
        # Disease network visualization
        if len(self.disease_network.nodes) > 0:
            try:
                # Calculate node positions using networkx
                pos = nx.spring_layout(self.disease_network, seed=42)
                
                # Create edges trace
                edge_x = []
                edge_y = []
                
                for edge in self.disease_network.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
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
            except Exception:
                pass # Skip if network visualization can't be generated
        
        return figures

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
    st.set_page_config(page_title="Medical Report Analyzer", layout="wide")
    
    st.title("🏥 Medical Report Analyzer")
    st.markdown("Upload medical reports for AI-powered analysis and predictions")
    
    # Sidebar for app info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This tool analyzes medical reports to:
        - Track disease progression
        - Identify risk factors
        - Predict potential future health risks
        - Analyze lab result trends
        
        *Supported File Type:*
        - Text files (.txt)
        
        *Note:* All analysis is for informational purposes only and should be verified by healthcare professionals.
        """)
        
        # Add reset button
        if st.button("Reset Analyzer"):
            if 'analyzer' in st.session_state:
                del st.session_state.analyzer
                st.success("Analyzer reset successfully!")
                st.rerun()
    
    # Initialize the analyzer
    if 'analyzer' not in st.session_state:
        try:
            st.session_state.analyzer = MedicalReportAnalyzer()
        except Exception as e:
            st.error(f"Error initializing analyzer: {str(e)}")
            st.error("Make sure all dependencies are installed correctly.")
            st.info("To install dependencies, run: `pip install streamlit pandas numpy spacy plotly scipy networkx`")
            return
    
    # Create tabs for different functions
    tab1, tab2, tab3 = st.tabs(["Upload Reports", "Analysis & Predictions", "Demo"])
    
    with tab1:
        st.header("Upload Medical Reports")
        st.caption("Note: Currently only supporting .txt files for simplified processing")
        uploaded_files = st.file_uploader(
            "Upload Medical Reports",
            type=['txt'],
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
                    else:
                        st.info("No significant disease predictions found with the current data.")
            
            with col2:
                st.subheader("Disease Progression")
                timeline_data = []
                for date in sorted(st.session_state.analyzer.temporal_data.keys()):
                    for disease in st.session_state.analyzer.temporal_data[date]['diseases']:
                        timeline_data.append({
                            'date': date,
                            'disease': disease
                        })
                
                if timeline_data:
                    st.dataframe(pd.DataFrame(timeline_data), use_container_width=True)
                else:
                    st.info("No disease progression data available.")
            
            st.subheader("Visualizations")
            figures = st.session_state.analyzer.generate_visualization()
            if figures:
                tabs = st.tabs([f"Chart {i+1}" for i in range(len(figures))])
                for i, tab in enumerate(tabs):
                    with tab:
                        st.plotly_chart(figures[i], use_container_width=True)
            else:
                st.info("No visualizations could be generated with the current data.")
    
    with tab3:
        st.header("Try Demo Data")
        st.markdown("Use sample medical reports to see how the analyzer works.")
        
        if st.button("Load Demo Data"):
            st.session_state.analyzer = MedicalReportAnalyzer()
            
            sample_data = create_sample_data()
            progress_bar = st.progress(0)
            
            for i, (filename, content) in enumerate(sample_data.items()):
                # Create a BytesIO object with the content
                file_obj = io.BytesIO(content.encode())
                file_obj.name = filename
                
                # Process the sample file
                date_str = re.search(r'Date:\s*(\d{4}-\d{2}-\d{2})', content)
                if date_str:
                    report_date = datetime.strptime(date_str.group(1), '%Y-%m-%d')
                else:
                    report_date = datetime.now()
                
                try:
                    st.session_state.analyzer.process_report(content, report_date, filename)
                    st.success(f"Successfully processed {filename}")
                except Exception as e:
                    st.error(f"Error processing {filename}: {str(e)}")
                
                # Update progress bar
                progress_bar.progress((i + 1) / len(sample_data))
            
            st.success("Demo data loaded! Go to the Analysis tab to see results.")

if __name__ == "__main__":
    main()