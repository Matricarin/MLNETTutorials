﻿using Microsoft.ML.Data;

namespace SentimentAnalysis
{
    public class SentimentData
    {
        [LoadColumn(0)]
        public string? SentimentText;
        [LoadColumn(1), ColumnName("label")]
        public bool Sentiment;
    }

    public class SentimentPrediction : SentimentData
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
        public float Score {get; set; }
    }
}