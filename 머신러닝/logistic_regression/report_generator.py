import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

class ReportGenerator:
    def __init__(self, symbol, data, features, predictions, probabilities, importance, eval_results):
        self.symbol = symbol
        self.data = data
        self.features = features
        self.predictions = predictions
        self.probabilities = probabilities
        self.importance = importance
        self.eval_results = eval_results
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=18,
            spaceAfter=20
        )
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10
        )
    
    def _create_volatility_chart(self):
        """변동성 차트를 생성합니다."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['Volatility'],
            name='변동성',
            line=dict(color='blue')
        ))
        fig.update_layout(
            title='주가 변동성 추이',
            xaxis_title='날짜',
            yaxis_title='변동성',
            height=500,
            template='plotly_white'
        )
        
        # Plotly 차트를 이미지로 변환
        img_bytes = fig.to_image(format="png")
        img_buffer = io.BytesIO(img_bytes)
        return Image(img_buffer, width=6*inch, height=4*inch)
    
    def _create_prediction_chart(self):
        """예측 확률 차트를 생성합니다."""
        test_indices = self.features.index[-len(self.predictions):]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_indices,
            y=self.probabilities,
            name='변동성 급증 확률',
            line=dict(color='red')
        ))
        fig.update_layout(
            title='변동성 급증 예측 확률',
            xaxis_title='날짜',
            yaxis_title='확률',
            height=500,
            template='plotly_white'
        )
        
        # Plotly 차트를 이미지로 변환
        img_bytes = fig.to_image(format="png")
        img_buffer = io.BytesIO(img_bytes)
        return Image(img_buffer, width=6*inch, height=4*inch)
    
    def _create_importance_chart(self):
        """특성 중요도 차트를 생성합니다."""
        if not self.importance.empty:
            fig = px.bar(
                self.importance,
                x='feature',
                y='importance',
                title='특성 중요도',
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                height=500,
                template='plotly_white'
            )
            
            # Plotly 차트를 이미지로 변환
            img_bytes = fig.to_image(format="png")
            img_buffer = io.BytesIO(img_bytes)
            return Image(img_buffer, width=6*inch, height=4*inch)
        return None
    
    def _create_performance_table(self):
        """모델 성능 테이블을 생성합니다."""
        data = [
            ['정확도', f"{float(self.eval_results['accuracy']):.3f}"],
            ['분류 보고서', self.eval_results['classification_report']]
        ]
        
        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        return table
    
    def generate_report(self, output_path):
        """PDF 보고서를 생성합니다."""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # 보고서 내용 생성
        story = []
        
        # 제목
        title = Paragraph(f"{self.symbol} 주식 변동성 예측 분석 보고서", self.title_style)
        story.append(title)
        story.append(Spacer(1, 12))
        
        # 생성 날짜
        date = Paragraph(f"생성 날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.body_style)
        story.append(date)
        story.append(Spacer(1, 12))
        
        # 모델 성능
        story.append(Paragraph("모델 성능", self.heading_style))
        story.append(self._create_performance_table())
        story.append(Spacer(1, 12))
        
        # 변동성 차트
        story.append(Paragraph("주가 변동성 추이", self.heading_style))
        story.append(self._create_volatility_chart())
        story.append(Spacer(1, 12))
        
        # 예측 확률 차트
        story.append(Paragraph("변동성 급증 예측 확률", self.heading_style))
        story.append(self._create_prediction_chart())
        story.append(Spacer(1, 12))
        
        # 특성 중요도 차트
        story.append(Paragraph("특성 중요도", self.heading_style))
        importance_chart = self._create_importance_chart()
        if importance_chart:
            story.append(importance_chart)
        else:
            story.append(Paragraph("특성 중요도를 계산할 수 없습니다.", self.body_style))
        story.append(Spacer(1, 12))
        
        # 데이터 요약
        story.append(Paragraph("데이터 요약", self.heading_style))
        data_summary = [
            ['데이터 포인트', str(len(self.features))],
            ['시작일', self.data.index[0].strftime('%Y-%m-%d')],
            ['종료일', self.data.index[-1].strftime('%Y-%m-%d')]
        ]
        
        summary_table = Table(data_summary, colWidths=[2*inch, 4*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        
        # PDF 생성
        doc.build(story)
        
        return output_path 