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
import warnings
warnings.filterwarnings('ignore')

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
    def __init__(self):
        self.reports = []
        self.temporal_data = {}
        self.disease_network = nx.Graph()

        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import os
            os.system("python -m spacy download en_core_web_sm")
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
            
            return self.process_report(content, report_date)
            
        except Exception as e:
            raise ValueError(f"Error processing file {file.name}: {str(e)}")

    def process_report(self, report_content, report_date):
        """Process individual report with temporal tracking"""
        doc = self.nlp(report_content.lower())
        
        report_data = {
            'date': report_date,
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
            if score > 1.0:
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
                correlation, p_value = pearsonr(
                    range(len(data['values'])),
                    data['values']
                )
                results[lab] = {
                    'trend': 'increasing' if correlation > 0 else 'decreasing',
                    'correlation': correlation,
                    'significance': p_value
                }
        
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
            fig = px.timeline(
                df,
                x_start='date',
                x_end='date',
                y='disease',
                title='Disease Progression Timeline'
            )
            figures.append(fig)
        
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
        
        return figures

def main():
    st.set_page_config(page_title="Predictive Medical Analyzer", layout="wide")
    
    st.title("🏥 Predictive Medical Report Analyzer")
    st.markdown("Upload medical reports for predictive analysis")
    
    # Initialize the analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = MedicalReportAnalyzer()
    
    # File upload with dates
    uploaded_files = st.file_uploader(
        "Upload Medical Reports",
        type=['txt', 'pdf', 'doc', 'docx'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            # Add date input for each file
            report_date = st.date_input(
                f"Date for {file.name}",
                datetime.now()
            )
            
            try:
                # Process file with proper error handling
                st.session_state.analyzer.process_file(file, report_date)
                st.success(f"Successfully processed {file.name}")
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                continue
        
        if st.button("Generate Predictive Analysis"):
            st.subheader("Predicted Future Health Risks")
            predictions = st.session_state.analyzer.predict_future_diseases()
            
            if predictions:
                for pred in predictions:
                    with st.expander(f"🔍 {pred['disease'].title()} - Risk Score: {pred['risk_score']:.2f}"):
                        st.write("*Contributing Factors:*")
                        for factor in pred['contributing_factors']:
                            if factor['type'] == 'risk_factor':
                              st.write(f"- {factor['category'].title()}: {factor['factor']}")
                            else:
                                st.write(f"- Lab {factor['lab']}: {factor['trend']} trend")
            else:
                st.info("No significant disease risks predicted based on current data.")
            
            # Show visualizations
            st.subheader("Analysis Visualizations")
            figures = st.session_state.analyzer.generate_visualization()
            for fig in figures:
                st.plotly_chart(fig, use_container_width=True)
            
            # Download full report
            report = {
                'predictions': predictions,
                'temporal_analysis': {
                    str(date): {
                        'diseases': list(data['diseases']),
                        'risk_factors': list(data['risk_factors']),
                        'lab_trends': {k: v for k, v in data['lab_trends'].items()}
                    }
                    for date, data in st.session_state.analyzer.temporal_data.items()
                },
                'generated_date': datetime.now().isoformat()
            }
            
            st.download_button(
                "📥 Download Full Analysis Report",
                json.dumps(report, indent=2),
                file_name="predictive_medical_analysis.json",
                mime="application/json"
            )
    
    # Add sidebar with information
    with st.sidebar:
        st.header("ℹ About")
        st.markdown("""
        This tool analyzes medical reports to:
        - Track disease progression
        - Identify risk factors
        - Predict potential future health risks
        - Analyze lab result trends
        
        *Supported File Types:*
        - Text files (.txt)
        - PDF documents (.pdf)
        - Word documents (.doc, .docx)
        
        *Note:* All analysis is for informational purposes only and should be verified by healthcare professionals.
        """)
        
        st.header("🔧 Settings")
        st.markdown("Future updates will include customizable analysis parameters.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>🏥 For educational and assistive purposes only. Always consult healthcare professionals for medical advice.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()