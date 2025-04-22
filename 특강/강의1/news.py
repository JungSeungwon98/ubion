
# 뉴스 수집 함수 정의
def get_naver_news(query, pages=1):
    print(f"[INFO] 뉴스 수집 시작: 검색어='{query}', 페이지 수={pages}")
    results = []
    for page in range(1, pages * 10, 10):
        target_url = f"https://search.naver.com/search.naver?where=news&query={query}&start={page}"
        print(f"[DEBUG] 요청 URL: {target_url}")
        encoded_target_url = quote(target_url,  encoding="utf-8")
        print(f"[DEBUG] 요청 URL: {encoded_target_url}")
        url = f"http://api.scrape.do?token=0ad8564c569f4ae0a8ace68e476519e9310be931402&url={encoded_target_url}"
        print(f"[DEBUG] 요청 URL: {url}")
        
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'})
        if response.status_code != 200:
            print(f"[ERROR] 요청 실패: 상태 코드 {response.status_code}")
            continue
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("div.Ermefm6A3ilpd9Zvt0OZ")
        print(f"[DEBUG] 페이지 {page//10 + 1} - 수집된 기사 수: {len(articles)}")
        for article in articles:
            title_tag = article.select_one("a.jT1DuARpwIlNAFMacxlu")
            info_group = article.select("div.pMazRkOPoM49VLcrhtmI a")
            if len(info_group) >= 2:
                url = info_group[1]['href']  # 두 번째 a 태그의 href 속성
            else:
                url = title_tag['href']  # 예비: 만약 두 번째 a 태그 없으면 기존 방식 사용
            title = title_tag.get_text()
            print(f"[DEBUG] 기사 제목: {title} | 원문 URL: {url}")
            content = get_article_content(url)
            results.append({"title": title, "url": url, "content": content})
    print(f"[INFO] 뉴스 수집 완료! 총 수집 건수: {len(results)}")
    return pd.DataFrame(results)