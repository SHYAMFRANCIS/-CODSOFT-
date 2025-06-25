import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class MovieGenrePredictor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.feature_names = []
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def create_sample_data(self):
        """Create sample movie dataset for demonstration"""
        movies_data = {
            'plot': [
                "A young wizard discovers his magical heritage and attends a school of wizardry where he learns about friendship, courage, and the battle between good and evil.",
                "In a dystopian future, a computer hacker discovers that reality is actually a simulation and joins a rebellion against the machines controlling humanity.",
                "Two detectives investigate a series of murders linked by the seven deadly sins, leading them into a dark psychological game with a serial killer.",
                "A group of friends reunite for a wedding, leading to romantic complications, misunderstandings, and ultimately finding true love.",
                "An alien spacecraft lands on Earth, and a linguist must learn to communicate with the extraterrestrial visitors to prevent global warfare.",
                "A superhero with incredible strength must save the city from a villain who threatens to destroy everything he holds dear.",
                "A young woman falls in love with a mysterious man, but their romance is complicated by supernatural forces and ancient curses.",
                "A group of scientists discover a way to travel through time, but their experiments have unintended consequences that threaten the fabric of reality.",
                "A detective investigates a murder case that leads him into the criminal underworld, where corruption and betrayal lurk around every corner.",
                "A family struggles to survive during a zombie apocalypse, fighting both the undead and other survivors in a post-apocalyptic world.",
                "A brilliant but eccentric inventor creates a machine that can predict the future, leading to unexpected consequences and moral dilemmas.",
                "Two rival gangs fight for control of the city streets in a violent conflict that spans generations and tests family loyalties.",
                "A young girl discovers she has magical powers and must learn to control them while navigating the challenges of adolescence and first love.",
                "A team of astronauts embarks on a dangerous mission to Mars, facing technical failures, personal conflicts, and the vast emptiness of space.",
                "A serial killer terrorizes a small town, and a federal agent must use psychological profiling to catch the murderer before more people die.",
                "A group of friends go on a camping trip that turns into a nightmare when they encounter a family of cannibalistic killers in the woods.",
                "A struggling actor finally gets his big break in Hollywood, but success comes at a personal cost as he loses touch with his roots and relationships.",
                "A historical epic about a warrior who leads his people in a rebellion against an oppressive empire, featuring massive battle scenes and political intrigue.",
                "A comedian's stand-up routine becomes the basis for a hilarious misunderstanding that snowballs into an absurd series of events.",
                "A sports team of underdogs must overcome personal differences and external challenges to win the championship and prove their worth."
            ],
            'genre': [
                'Fantasy', 'Sci-Fi', 'Thriller', 'Romance', 'Sci-Fi',
                'Action', 'Romance', 'Sci-Fi', 'Crime', 'Horror',
                'Sci-Fi', 'Action', 'Fantasy', 'Sci-Fi', 'Thriller',
                'Horror', 'Drama', 'Action', 'Comedy', 'Drama'
            ]
        }
        return pd.DataFrame(movies_data)
    
    def extract_features(self, texts, fit=True):
        """Extract TF-IDF features from text data"""
        if fit:
            features = self.tfidf_vectorizer.fit_transform(texts)
            self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        else:
            features = self.tfidf_vectorizer.transform(texts)
        return features
    
    def train_models(self, X_train, y_train):
        """Train multiple classification models"""
        # Define models
        models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        # Train each model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(results[name]['classification_report'])
        
        return results
    
    def predict_genre(self, plot_text, model_name='Logistic Regression'):
        """Predict genre for a new movie plot"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        # Preprocess the text
        processed_text = self.preprocess_text(plot_text)
        
        # Extract features
        features = self.tfidf_vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.models[model_name].predict(features)[0]
        probabilities = self.models[model_name].predict_proba(features)[0]
        
        # Get class labels
        classes = self.models[model_name].classes_
        
        # Create probability dictionary
        prob_dict = dict(zip(classes, probabilities))
        
        return prediction, prob_dict
    
    def plot_feature_importance(self, model_name='Logistic Regression', top_n=20):
        """Plot feature importance for interpretability"""
        if model_name not in self.models:
            return
        
        model = self.models[model_name]
        
        if hasattr(model, 'coef_'):
            # For linear models
            if len(model.classes_) > 2:
                # Multi-class: take mean of absolute coefficients
                importance = np.mean(np.abs(model.coef_), axis=0)
            else:
                # Binary: use coefficients directly
                importance = np.abs(model.coef_[0])
            
            # Get top features
            top_indices = importance.argsort()[-top_n:][::-1]
            top_features = [self.feature_names[i] for i in top_indices]
            top_importance = importance[top_indices]
            
            # Plot
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_importance)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Features - {model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
    
    def generate_wordcloud(self, texts, genres):
        """Generate word clouds for different genres"""
        unique_genres = np.unique(genres)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, genre in enumerate(unique_genres[:6]):  # Limit to 6 genres for visualization
            if i >= len(axes):
                break
                
            # Get texts for this genre
            genre_texts = [texts[j] for j in range(len(texts)) if genres[j] == genre]
            combined_text = ' '.join(genre_texts)
            
            # Generate word cloud
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate(combined_text)
            
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'{genre} Genre')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(unique_genres), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def cross_validate_models(self, X, y, cv=5):
        """Perform cross-validation for all models"""
        cv_results = {}
        
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            cv_results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
            
            print(f"{name} - CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results

# Example usage and demonstration
def main():
    # Initialize the predictor
    predictor = MovieGenrePredictor()
    
    # Create sample data (in practice, you would load your own dataset)
    print("Creating sample movie dataset...")
    df = predictor.create_sample_data()
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Genres: {df['genre'].value_counts()}")
    
    # Preprocess text data
    print("\nPreprocessing text data...")
    df['processed_plot'] = df['plot'].apply(predictor.preprocess_text)
    
    # Extract features
    print("Extracting TF-IDF features...")
    X = predictor.extract_features(df['processed_plot'])
    y = df['genre']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train models
    print("\nTraining models...")
    predictor.train_models(X_train, y_train)
    
    # Evaluate models
    print("\nEvaluating models...")
    results = predictor.evaluate_models(X_test, y_test)
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_results = predictor.cross_validate_models(X, y)
    
    # Test prediction on new data
    print("\nTesting prediction on new movie plot...")
    new_plot = """A young woman discovers she has the ability to time travel and must use her powers 
                  to prevent a catastrophic event that threatens to destroy the world."""
    
    predicted_genre, probabilities = predictor.predict_genre(new_plot)
    print(f"Predicted genre: {predicted_genre}")
    print("Genre probabilities:")
    for genre, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {genre}: {prob:.4f}")
    
    # Feature importance visualization
    try:
        predictor.plot_feature_importance('Logistic Regression')
    except:
        print("Feature importance plot not available")
    
    # Word cloud visualization
    try:
        predictor.generate_wordcloud(df['processed_plot'].tolist(), df['genre'].tolist())
    except:
        print("Word cloud generation not available")

if __name__ == "__main__":
    main()