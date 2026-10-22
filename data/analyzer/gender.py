import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from analyzer.constants import DATA_DIR

class GenderDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        # Internal database - never displayed
        self.name_database = {
            'male': [
                'abhi', 'anmol', 'anshveer', 'anuj', 'armaandeep',
                'arnav', 'arun', 'ayush', 'biplove', 'devansh',
                'gaganpreet', 'gurbaaj', 'gurbaksh', 'gurkanwar',
                'gurkirat', 'gurnoor', 'gurpreet', 'gursimar',
                'gurtej', 'hardaksh', 'harmanpreet', 'harsimrat',
                'ishan', 'karan', 'karanveer', 'keshav', 'krishan',
                'mankaran', 'manmeet', 'mauli', 'md sahil', 'navjot',
                'nikhil', 'prabpreet', 'rakesh', 'reetinder', 'rishu',
                'sahibpreet', 'sahil', 'samarveer', 'samridh',
                'sanyam', 'sharanjit', 'shubh', 'snimerbir',
                'snover', 'subham', 'swarnava', 'tajvir', 'tamish',
                'tripanjeet', 'udeybir', 'varun', 'vikramjeet',
                'vinay', 'gurshaan', 'arshdeep', 'dipanshu',
                'harman', 'jasnoor', 'manish', 'parmeet',
                'prabhjot', 'rhythm', 'prabhsimran'
            ],
            'female': [
                'akriti', 'alpana', 'bhavya', 'chanpreet', 'chetna',
                'gunsimranjot', 'gurunaamdeep', 'harmandeep', 'harpreet',
                'harsimar', 'navneet', 'nisha', 'prabhleen', 'rashi',
                'rajwant', 'ramandeep', 'ravinder', 'simran',
                'sukhmeet', 'taniya', 'tanu', 'anshika', 'priya',
                'neha', 'pooja', 'anjali', 'sneha', 'divya', 'isha',
                'riya', 'shreya', 'kajal', 'jasmine'
            ]
        }
        self.suffix_patterns = {
            'male': ['singh', 'kumar', 'sharma', 'verma', 'gupta', 'jain', 
                    'agarwal', 'malhotra', 'mehra', 'kapoor', 'khanna', 'chopra',
                    'arora', 'gill', 'sodhi', 'galib', 'sahni', 'sofat', 
                    'sehgal', 'bhangu', 'rattu', 'mourya', 'bhaskar', 'ahuja',
                    'pal', 'thakur', 'tagger', 'yadav', 'patel', 'shah',
                    'joshi', 'pandey', 'tiwari', 'dubey', 'mishra'],
            'female': ['kaur', 'devi', 'rani', 'kumari']
        }
        self.training_count = {
            'male': len(self.name_database['male']),
            'female': len(self.name_database['female'])
        }
        # Auto-train on initialization if model doesn't exist
        if not os.path.exists(str(DATA_DIR / 'gender_model.pkl')):
            self.auto_train()
        else:
            self.load_model(str(DATA_DIR / 'gender_model.pkl'))
    
    def auto_train(self):
        """Automatically train the model with existing data"""
        try:
            accuracy = self.train_model()
            if accuracy:
                self.save_model(str(DATA_DIR / 'gender_model.pkl'))
            return accuracy
        except:
            return None
    
    def detect_gender(self, name):
        """Automatically detect gender - no names displayed"""
        if pd.isna(name) or name == '':
            return 'Unknown'
        
        name = str(name).strip().lower()
        # Clean the name
        name = name.replace('/', '').replace('\\', '').strip()
        
        # Split name to get first name for better detection
        name_parts = name.split()
        first_name = name_parts[0] if name_parts else name
        
        # Method 1: Check in database (check full name and first name)
        for gender, names in self.name_database.items():
            if name in names or first_name in names:
                return 'Male' if gender == 'male' else 'Female'
        
        # Method 2: Check for 'KAUR' (Female) vs 'SINGH' (Male)
        if 'kaur' in name:
            return 'Female'
        if 'singh' in name:
            # Check if it's a female name with Singh
            for female in self.name_database['female']:
                if female in name or female in first_name:
                    return 'Female'
            return 'Male'
        
        # Method 3: Check for suffixes
        for gender, suffixes in self.suffix_patterns.items():
            for suffix in suffixes:
                if name.endswith(suffix) or first_name.endswith(suffix):
                    return 'Male' if gender == 'male' else 'Female'
        
        # Method 4: Pattern recognition - Female names often end with 'a', 'i', 'e'
        if first_name.endswith('a') or first_name.endswith('i') or first_name.endswith('e'):
            male_exceptions = ['arjun', 'karan', 'mohan', 'rahul', 'ankit', 'vikram', 
                              'raj', 'amit', 'suresh', 'ramesh', 'mahesh']
            if first_name not in male_exceptions:
                if first_name not in self.name_database['male']:
                    return 'Female'
        
        # Method 5: First letter patterns
        male_first_letters = ['a', 'b', 'd', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'r', 's', 't', 'u', 'v']
        
        if first_name:
            first_letter = first_name[0]
            
            if first_letter in male_first_letters:
                # Names ending with 'eet' or 'een' or 'aj' are often male
                if first_name.endswith('eet') or first_name.endswith('een') or first_name.endswith('aj'):
                    return 'Male'
                
                # Check for male name endings
                male_endings = ['bir', 'jeet', 'preet', 'deep', 'inder', 'veer', 'pal']
                for ending in male_endings:
                    if first_name.endswith(ending):
                        return 'Male'
        
        # Method 6: Use trained model if available
        if self.model is not None and self.vectorizer is not None:
            try:
                name_vectorized = self.vectorizer.transform([name])
                prediction = self.model.predict(name_vectorized)
                return prediction[0]
            except:
                pass
        
        return 'Unknown'
    
    def train_model(self):
        """Train the machine learning model"""
        names = []
        genders = []
        
        # Add existing database
        for gender, names_list in self.name_database.items():
            for name in names_list:
                names.append(name)
                genders.append('Male' if gender == 'male' else 'Female')
        
        # Add common patterns
        male_suffixes = ['singh', 'kumar', 'sharma', 'verma', 'gupta', 'jain',
                        'yadav', 'patel', 'shah', 'joshi', 'pandey']
        for suffix in male_suffixes:
            names.append(suffix)
            genders.append('Male')
        
        female_suffixes = ['kaur', 'devi', 'kumari']
        for suffix in female_suffixes:
            names.append(suffix)
            genders.append('Female')
        
        # Create feature vectors
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 4))
        X = self.vectorizer.fit_transform(names)
        y = np.array(genders)
        
        # Split data
        if len(np.unique(y)) >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            self.model = MultinomialNB()
            self.model.fit(X_train, y_train)
            
            # Calculate accuracy
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            return accuracy
        return None
    
    def save_model(self, filepath):
        """Save the trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'database': self.name_database,
                'suffixes': self.suffix_patterns,
                'training_count': self.training_count
            }, f)
    
    def load_model(self, filepath):
        """Load a trained model"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.vectorizer = data['vectorizer']
                    self.name_database = data['database']
                    self.suffix_patterns = data['suffixes']
                    self.training_count = data.get('training_count', 
                        {'male': len(self.name_database['male']), 
                         'female': len(self.name_database['female'])})
                return True
            except:
                return False
        return False
    
    def get_stats(self):
        """Get statistics without exposing names"""
        return {
            'male_count': self.training_count['male'],
            'female_count': self.training_count['female'],
            'total': self.training_count['male'] + self.training_count['female'],
            'model_trained': self.model is not None
        }

# Initialize the gender detector once per server process
@st.cache_resource
def get_gender_detector():
    return GenderDetector()

