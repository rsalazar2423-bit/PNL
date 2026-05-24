import joblib
import pandas as pd
import os
import gc

def migrate_cache():
    if not os.path.exists('pipeline_results_data.pkl'):
        print("No old cache found.")
        return
        
    print("Loading old monolithic cache...")
    data = joblib.load('pipeline_results_data.pkl')
    
    df = data['df']
    
    print(f"Saving dataframe to pipeline_data.parquet ({len(df)} rows)...")
    df.to_parquet('pipeline_data.parquet', index=False)
    
    print("Saving ML models to pipeline_models.pkl...")
    models = {
        'tfidf': data.get('tfidf'),
        'ner_pos': data.get('ner_pos'),
        'cluster_results': data.get('cluster_results'),
        'rag': data.get('rag')
    }
    joblib.dump(models, 'pipeline_models.pkl')
    
    print("Migration complete!")

if __name__ == "__main__":
    migrate_cache()
