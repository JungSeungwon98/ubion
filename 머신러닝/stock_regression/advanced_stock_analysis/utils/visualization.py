import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드를 사용하지 않도록 설정
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Union, Optional
import logging
import os
from datetime import datetime

class StockVisualizer:
    def __init__(self, style: str = 'seaborn'):
        """
        주식 데이터 시각화 클래스
        
        Parameters:
        -----------
        style : str
            matplotlib 스타일 ('seaborn', 'dark_background', 'bmh' 등)
        """
        plt.style.use(style)
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def plot_price_history(self, data: pd.DataFrame, save_path: Optional[str] = None):
        """가격 히스토리 플롯"""
        fig = plt.figure(figsize=(15, 10))
        
        # 주가 및 거래량 서브플롯
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        
        # 주가 차트
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(data.index, data['Close'], label='Close Price')
        
        # 필요한 컬럼이 있는지 확인하고 플롯
        if 'SMA_20' in data.columns:
            ax1.plot(data.index, data['SMA_20'], label='20-day SMA', alpha=0.7)
        if 'SMA_50' in data.columns:
            ax1.plot(data.index, data['SMA_50'], label='50-day SMA', alpha=0.7)
        if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
            ax1.fill_between(data.index, data['BB_Upper'], data['BB_Lower'],
                            alpha=0.2, label='Bollinger Bands')
        
        ax1.set_title('Stock Price History with Technical Indicators')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # 거래량 차트
        ax2 = fig.add_subplot(gs[1])
        ax2.bar(data.index, data['Volume'], alpha=0.7)
        ax2.set_title('Trading Volume')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_technical_indicators(self, data: pd.DataFrame,
                                save_path: Optional[str] = None):
        """기술적 지표 플롯"""
        fig = plt.figure(figsize=(15, 15))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1])
        
        # 주가 및 이동평균선
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(data.index, data['Close'], label='Close Price')
        
        # 필요한 컬럼이 있는지 확인하고 플롯
        if 'SMA_20' in data.columns:
            ax1.plot(data.index, data['SMA_20'], label='20-day SMA')
        if 'SMA_50' in data.columns:
            ax1.plot(data.index, data['SMA_50'], label='50-day SMA')
        if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
            ax1.fill_between(data.index, data['BB_Upper'], data['BB_Lower'],
                            alpha=0.2, label='Bollinger Bands')
        
        ax1.set_title('Price and Moving Averages')
        ax1.legend()
        ax1.grid(True)
        
        # RSI
        ax2 = fig.add_subplot(gs[1])
        if 'RSI' in data.columns:
            ax2.plot(data.index, data['RSI'])
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax2.set_title('RSI')
        else:
            ax2.text(0.5, 0.5, 'RSI data not available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes)
            ax2.set_title('RSI (Not Available)')
        ax2.grid(True)
        
        # MACD
        ax3 = fig.add_subplot(gs[2])
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns and 'MACD_Hist' in data.columns:
            ax3.plot(data.index, data['MACD'], label='MACD')
            ax3.plot(data.index, data['MACD_Signal'], label='Signal')
            ax3.bar(data.index, data['MACD_Hist'], alpha=0.3, label='Histogram')
            ax3.set_title('MACD')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'MACD data not available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes)
            ax3.set_title('MACD (Not Available)')
        ax3.grid(True)
        
        # Stochastic
        ax4 = fig.add_subplot(gs[3])
        if 'Stoch_K' in data.columns and 'Stoch_D' in data.columns:
            ax4.plot(data.index, data['Stoch_K'], label='%K')
            ax4.plot(data.index, data['Stoch_D'], label='%D')
            ax4.axhline(y=80, color='r', linestyle='--', alpha=0.5)
            ax4.axhline(y=20, color='g', linestyle='--', alpha=0.5)
            ax4.set_title('Stochastic Oscillator')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Stochastic data not available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes)
            ax4.set_title('Stochastic Oscillator (Not Available)')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_returns_analysis(self, df, output_path='returns_analysis.html'):
        """수익률 분석 차트 생성 (HTML)"""
        try:
            # 데이터 확인
            if df is None or df.empty:
                self.logger.error("DataFrame is empty, cannot create returns analysis")
                return None
            
            if 'Returns' not in df.columns:
                self.logger.error("Missing Returns column for returns analysis")
                return None
            
            # plotly 사용
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # 서브플롯 생성
            fig = make_subplots(rows=2, cols=2,
                              subplot_titles=(
                                  'Returns Distribution', 
                                  'Cumulative Returns',
                                  'Rolling Volatility',
                                  'Returns QQ Plot'
                              ))
            
            # 1. 수익률 분포 히스토그램
            returns = df['Returns'].dropna()
            
            if len(returns) == 0:
                self.logger.error("No valid returns data after dropping NaN values")
                return None
            
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    name='Returns Distribution',
                    marker_color='blue',
                    opacity=0.7,
                    nbinsx=30
                ),
                row=1, col=1
            )
            
            # 2. 누적 수익률
            cumulative_returns = (1 + returns).cumprod() - 1
            
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns,
                    name='Cumulative Returns',
                    line=dict(color='green')
                ),
                row=1, col=2
            )
            
            # 3. 롤링 변동성 (Rolling Volatility)
            if len(returns) >= 21:  # 최소 21일 필요
                rolling_vol = returns.rolling(window=21).std() * np.sqrt(252)  # 연간화
                
                fig.add_trace(
                    go.Scatter(
                        x=rolling_vol.index,
                        y=rolling_vol,
                        name='21-day Rolling Volatility (Annualized)',
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )
            
            # 4. QQ Plot (Quantile-Quantile Plot)
            from scipy import stats
            
            # 이론적 분위수
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
            
            # 실제 수익률 분위수
            returns_sorted = sorted(returns.dropna())
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=returns_sorted,
                    mode='markers',
                    name='QQ Plot',
                    marker=dict(color='purple')
                ),
                row=2, col=2
            )
            
            # 레퍼런스 라인 추가
            z = np.polyfit(theoretical_quantiles, returns_sorted, 1)
            y_hat = np.poly1d(z)(theoretical_quantiles)
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=y_hat,
                    mode='lines',
                    name='Normal Fit',
                    line=dict(color='red', dash='dash')
                ),
                row=2, col=2
            )
            
            # 레이아웃 조정
            fig.update_layout(
                title='Returns Analysis',
                height=800,
                template='plotly_white',
                showlegend=True
            )
            
            # HTML 파일로 저장
            fig.write_html(output_path)
            self.logger.info(f"Returns analysis saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating returns analysis: {str(e)}")
            return None
    
    def plot_correlation_matrix(self, df, output_path='correlation_matrix.html'):
        """상관관계 행렬 차트 생성 (HTML)"""
        try:
            # 데이터 확인
            if df is None or df.empty:
                self.logger.error("DataFrame is empty, cannot create correlation matrix")
                return None
            
            # 주요 컬럼만 선택 (최대 15개)
            main_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']
            # 기술적 지표 추가
            tech_cols = [col for col in df.columns if any(ind in col for ind in ['SMA', 'EMA', 'RSI', 'MACD', 'BB_'])]
            
            # 모든 컬럼 결합 및 제한
            all_cols = main_cols + tech_cols
            # DataFrame에 있는 컬럼만 필터링
            cols_to_use = [col for col in all_cols if col in df.columns]
            
            # 너무 많은 컬럼이면 제한
            if len(cols_to_use) > 15:
                self.logger.warning(f"Too many columns ({len(cols_to_use)}), limiting to 15")
                cols_to_use = cols_to_use[:15]
            
            # 상관관계 계산
            corr_matrix = df[cols_to_use].corr()
            
            # plotly 사용
            import plotly.graph_objects as go
            
            # 히트맵 생성
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.index,
                y=corr_matrix.columns,
                colorscale='RdBu_r',  # Red-Blue 스케일 (반전)
                zmid=0,  # 0을 중심으로 색상 스케일
                colorbar=dict(title='Correlation')
            ))
            
            # 레이아웃 조정
            fig.update_layout(
                title='Correlation Matrix',
                xaxis_title='Features',
                yaxis_title='Features',
                height=800,
                width=800,
                template='plotly_white'
            )
            
            # HTML 파일로 저장
            fig.write_html(output_path)
            self.logger.info(f"Correlation matrix saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating correlation matrix: {str(e)}")
            return None
    
    def create_candlestick_chart(self, df, output_path='candlestick.html'):
        """캔들스틱 차트 생성 (HTML)"""
        try:
            # 데이터 확인
            if df is None or df.empty:
                self.logger.error("DataFrame is empty, cannot create candlestick chart")
                return None
            
            if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                self.logger.error("Missing OHLC columns for candlestick chart")
                return None
            
            # plotly 사용
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # 2행 1열 서브플롯 생성 (캔들스틱 + 거래량)
            fig = make_subplots(rows=2, cols=1, 
                              row_heights=[0.7, 0.3],
                              vertical_spacing=0.03,
                              subplot_titles=('Price', 'Volume'))
            
            # 캔들스틱 차트
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='OHLC'
                ),
                row=1, col=1
            )
            
            # SMA 추가 (있는 경우만)
            if 'SMA_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['SMA_20'],
                        name='SMA 20',
                        line=dict(color='orange')
                    ),
                    row=1, col=1
                )
            
            if 'SMA_50' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['SMA_50'],
                        name='SMA 50',
                        line=dict(color='purple')
                    ),
                    row=1, col=1
                )
            
            # 거래량 차트 (있는 경우만)
            if 'Volume' in df.columns:
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['Volume'],
                        name='Volume',
                        marker_color='rgba(0, 0, 255, 0.3)'
                    ),
                    row=2, col=1
                )
            
            # 레이아웃 조정
            fig.update_layout(
                title='Price History with Volume',
                xaxis_rangeslider_visible=False,
                height=600,
                template='plotly_white'
            )
            
            # x축 범위 일치시키기
            fig.update_xaxes(matches='x')
            
            # HTML 파일로 저장
            fig.write_html(output_path)
            self.logger.info(f"Candlestick chart saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating candlestick chart: {str(e)}")
            return None
    
    def create_volume_chart(self, df, output_path='volume.html'):
        """거래량 차트 생성 (HTML)"""
        try:
            # 데이터 확인
            if df is None or df.empty:
                self.logger.error("DataFrame is empty, cannot create volume chart")
                return None
            
            if 'Volume' not in df.columns:
                self.logger.error("Missing Volume column for volume chart")
                return None
            
            # plotly 사용
            import plotly.graph_objects as go
            
            # 거래량 차트
            fig = go.Figure()
            
            # 거래량 바 차트
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color='rgba(0, 0, 255, 0.5)'
                )
            )
            
            # 이동평균 추가 (있는 경우만)
            if 'Volume_SMA' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['Volume_SMA'],
                        name='Volume MA',
                        line=dict(color='red', width=2)
                    )
                )
            
            # 레이아웃 조정
            fig.update_layout(
                title='Trading Volume Analysis',
                xaxis_title='Date',
                yaxis_title='Volume',
                height=500,
                template='plotly_white'
            )
            
            # HTML 파일로 저장
            fig.write_html(output_path)
            self.logger.info(f"Volume chart saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating volume chart: {str(e)}")
            return None
    
    def create_technical_indicators_chart(self, df, indicators=None, output_path='indicators.html'):
        """기술적 지표 차트 생성 (HTML)"""
        try:
            # 데이터 확인
            if df is None or df.empty:
                self.logger.error("DataFrame is empty, cannot create indicators chart")
                return None
            
            # 지표가 없으면 자동으로 탐지
            if indicators is None:
                indicators = [col for col in df.columns if any(ind in col for ind in ['SMA', 'EMA', 'RSI', 'MACD'])]
            
            if not indicators:
                self.logger.error("No technical indicators found in DataFrame")
                return None
            
            # plotly 사용
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # 서브플롯 생성 (RSI, MACD, 기타 지표)
            fig = make_subplots(rows=3, cols=1, 
                              row_heights=[0.4, 0.3, 0.3],
                              vertical_spacing=0.08,
                              subplot_titles=('Price with MA', 'Momentum Indicators', 'Volume Indicators'))
            
            # 종가와 이동평균선
            if 'Close' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['Close'],
                        name='Close',
                        line=dict(color='black', width=1)
                    ),
                    row=1, col=1
                )
            
            # 이동평균선 추가
            for ma in [col for col in indicators if 'SMA' in col or 'EMA' in col]:
                if ma in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[ma],
                            name=ma,
                            line=dict(width=1)
                        ),
                        row=1, col=1
                    )
            
            # RSI 추가
            if 'RSI' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['RSI'],
                        name='RSI',
                        line=dict(color='blue')
                    ),
                    row=2, col=1
                )
                
                # 과매수/과매도 기준선
                fig.add_shape(
                    type="line", line_color="red", line_dash="dash",
                    x0=df.index[0], y0=70, x1=df.index[-1], y1=70,
                    row=2, col=1
                )
                fig.add_shape(
                    type="line", line_color="green", line_dash="dash",
                    x0=df.index[0], y0=30, x1=df.index[-1], y1=30,
                    row=2, col=1
                )
            
            # MACD 추가
            macd_cols = [col for col in df.columns if 'MACD' in col]
            if macd_cols:
                for macd_col in macd_cols:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[macd_col],
                            name=macd_col,
                        ),
                        row=2, col=1
                    )
            
            # 볼린저 밴드 추가
            bb_cols = [col for col in df.columns if 'BB_' in col]
            if bb_cols and len(bb_cols) >= 3:
                for bb_col in bb_cols:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[bb_col],
                            name=bb_col,
                            line=dict(dash='dot' if 'middle' not in bb_col else 'solid')
                        ),
                        row=3, col=1
                    )
            
            # 거래량 지표 추가
            vol_cols = [col for col in df.columns if 'Volume' in col and col != 'Volume']
            if vol_cols:
                for vol_col in vol_cols:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[vol_col],
                            name=vol_col
                        ),
                        row=3, col=1
                    )
            
            # 레이아웃 조정
            fig.update_layout(
                title='Technical Indicators Analysis',
                height=900,
                template='plotly_white',
                showlegend=True
            )
            
            # x축 범위 일치시키기
            fig.update_xaxes(matches='x')
            
            # HTML 파일로 저장
            fig.write_html(output_path)
            self.logger.info(f"Technical indicators chart saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating technical indicators chart: {str(e)}")
            return None
    
    def plot_feature_importance(self, feature_importance, top_n=10, model_name='Model', save_path=None):
        """특성 중요도 시각화"""
        try:
            # 중요도 정렬
            feature_importance = feature_importance.sort_values('importance', ascending=False).head(top_n)
            
            # 플롯 생성
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance)
            plt.title(f'{model_name} Feature Importance (Top {top_n})')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            # 저장 및 반환
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Feature importance plot saved to {save_path}")
            
            plt.close()
            return save_path
        
        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {str(e)}")
            return None
    
    def create_dashboard(self, df, output_dir='dashboard'):
        """주식 데이터 대시보드 생성"""
        try:
            self.logger.info(f"Creating dashboard in {output_dir}")
            
            # 데이터 유효성 검사
            if df is None or df.empty:
                self.logger.error("DataFrame is empty, cannot create dashboard")
                return None
            
            if len(df) < 5:  # 최소 데이터 포인트 확인
                self.logger.error(f"Insufficient data points ({len(df)}) for dashboard")
                return None
            
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 기본 정보 계산
            symbol = df.index.name if df.index.name else 'Stock'
            start_date = df.index.min().strftime('%Y-%m-%d')
            end_date = df.index.max().strftime('%Y-%m-%d')
            
            # 필수 열 존재 확인
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.logger.warning(f"Missing required columns: {missing_cols}")
                # 가능한 경우 열을 채움
                if 'Close' not in df.columns and 'Adj Close' in df.columns:
                    df['Close'] = df['Adj Close']
                    self.logger.info("Used 'Adj Close' for 'Close'")
            
            # 시각화 생성
            try:
                # 1. 주가 차트 (캔들스틱)
                if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                    self.create_candlestick_chart(df, output_path=os.path.join(output_dir, 'price_chart.html'))
                else:
                    self.logger.warning("Cannot create candlestick chart due to missing OHLC columns")
                
                # 2. 거래량 차트
                if 'Volume' in df.columns:
                    self.create_volume_chart(df, output_path=os.path.join(output_dir, 'volume_chart.html'))
                else:
                    self.logger.warning("Cannot create volume chart due to missing Volume column")
                
                # 3. 기술적 지표 차트 (있는 지표만)
                tech_indicators = [col for col in df.columns if any(ind in col for ind in ['SMA', 'EMA', 'RSI', 'MACD'])]
                if tech_indicators:
                    self.create_technical_indicators_chart(df, indicators=tech_indicators, 
                                                         output_path=os.path.join(output_dir, 'indicators_chart.html'))
                else:
                    self.logger.warning("No technical indicators found for chart")
                
                # 4. 수익률 분석
                if 'Returns' in df.columns:
                    self.plot_returns_analysis(df, output_path=os.path.join(output_dir, 'returns_analysis.html'))
                else:
                    self.logger.warning("Cannot create returns analysis due to missing Returns column")
                
                # 5. 상관관계 행렬
                self.plot_correlation_matrix(df, output_path=os.path.join(output_dir, 'correlation_matrix.html'))
            
            except Exception as e:
                self.logger.error(f"Error creating individual chart: {str(e)}")
                # 차트 일부만 생성되어도 계속 진행
            
            # 대시보드 HTML 파일 경로
            dashboard_path = os.path.join(output_dir, 'dashboard.html')
            
            # 대시보드 HTML 생성
            dashboard_html = self._create_dashboard_html(symbol, start_date, end_date)
            
            # HTML 파일로 저장
            with open(dashboard_path, 'w') as f:
                f.write(dashboard_html)
            
            self.logger.info(f"Dashboard created at {dashboard_path}")
            return dashboard_path
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard: {str(e)}")
            return None

    def _create_dashboard_html(self, symbol, start_date, end_date):
        """대시보드 HTML 생성"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{symbol} 주식 분석 대시보드</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            padding: 20px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }}
        .chart-container {{
            margin-bottom: 30px;
            border: 1px solid #eee;
            border-radius: 5px;
            padding: 15px;
        }}
        h1, h2 {{
            color: #333;
        }}
        .info {{
            color: #666;
            margin-bottom: 10px;
        }}
        iframe {{
            width: 100%;
            height: 500px;
            border: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{symbol} 주식 분석 대시보드</h1>
            <p class="info">분석 기간: {start_date} ~ {end_date}</p>
            <p class="info">생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="chart-container">
            <h2>가격 차트</h2>
            <iframe src="price_chart.html"></iframe>
        </div>
        
        <div class="chart-container">
            <h2>거래량 차트</h2>
            <iframe src="volume_chart.html"></iframe>
        </div>
        
        <div class="chart-container">
            <h2>기술적 지표</h2>
            <iframe src="indicators_chart.html"></iframe>
        </div>
        
        <div class="chart-container">
            <h2>수익률 분석</h2>
            <iframe src="returns_analysis.html"></iframe>
        </div>
        
        <div class="chart-container">
            <h2>상관관계 행렬</h2>
            <iframe src="correlation_matrix.html"></iframe>
        </div>
    </div>
</body>
</html>
        """
        return html_content

def main():
    # 예제 사용법
    import yfinance as yf
    from ..data.feature_engineering import FeatureEngineer
    
    # 데이터 가져오기
    symbol = 'AAPL'
    stock = yf.Ticker(symbol)
    data = stock.history(start='2020-01-01')
    
    # 특성 엔지니어링
    engineer = FeatureEngineer()
    data = engineer.add_technical_indicators(data)
    data = engineer.add_time_features(data)
    data = engineer.add_lagged_features(data)
    data = engineer.add_rolling_features(data)
    
    # 시각화
    visualizer = StockVisualizer()
    dashboard_path = visualizer.create_dashboard(data)
    print(f"Dashboard saved to: {dashboard_path}")

if __name__ == "__main__":
    main() 