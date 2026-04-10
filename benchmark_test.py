import os
import glob
import json
from src.chunking import RecursiveChunker, ChunkingStrategyComparator
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from src.models import Document
from src.embeddings import LocalEmbedder

def main():
    data_dir = "data"
    files = glob.glob(os.path.join(data_dir, "*.txt"))
    
    print(f"--- Đọc {len(files)} tài liệu ---")
    documents_raw = []
    
    for file_path in files:
        filename = os.path.basename(file_path).replace(".txt", "")
        parts = filename.split(" - ")
        title = parts[0].strip()
        author = parts[1].strip() if len(parts) > 1 else "Unknown"
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        documents_raw.append({"content": content, "title": title, "author": author, "file_path": file_path, "filename": filename})
        print(f"Đã nạp: '{title}' (Tác giả: {author}) - {len(content)} ký tự")
    
    print("\n--- Khởi tạo EmbeddingStore & Nạp dữ liệu ---")
    chunk_size = 700
    chunker = RecursiveChunker(chunk_size=chunk_size)
    
    docs_to_store = []
    for i, raw_doc in enumerate(documents_raw):
        chunks = chunker.chunk(raw_doc["content"])
        for chunk_idx, chunk_content in enumerate(chunks):
            doc = Document(
                id=f"{raw_doc['filename']}_chunk{chunk_idx}",
                content=chunk_content,
                metadata={"title": raw_doc["title"], "author": raw_doc["author"], "source": raw_doc["file_path"]}
            )
            docs_to_store.append(doc)
            
    store = EmbeddingStore(embedding_fn=LocalEmbedder(model_name="all-MiniLM-L6-v2"))
    store.add_documents(docs_to_store)
    num_chunks = store.get_collection_size()
    print(f"Tổng số chunks trong store: {num_chunks}")
    
    queries = [
        {
            "query_text": "Trong 'Anh đừng lỗi hẹn', tại sao Thuý Hằng lại ly dị chồng và nguyên nhân nào dẫn đến bệnh tim của cô?",
            "filter_dict": {"title": "Anh đừng lỗi hẹn"},
            "gold_answer": "Thuý Hằng ly dị vì chồng cô là người tẻ nhạt, tàn nhẫn, chỉ biết hưởng thụ và kiếm những đồng tiền khiến cô luôn cảm thấy lo âu. Chính những lo âu triền miên về những đồng tiền đó là nguyên nhân phát sinh bệnh tim của cô.",
            "source_file": "Anh đừng lỗi hẹn - Vũ Đức Nghĩa.txt"
        },
        {
            "query_text": "Nhân vật 'tôi' gặp người con trai trong truyện qua phương tiện nào?",
            "filter_dict": None,
            "gold_answer": "Nhân vật 'tôi' gặp anh qua blog — tình cờ vào blog của anh khi tha thẩn trên mạng, đọc những dòng nhật ký rồi để lại comment 'Xin cám ơn anh', từ đó trở thành bạn và dần thân thiết qua email và điện thoại.",
            "source_file": "48 giờ yêu nhau - Hà Thanh Phúc.txt"
        },
        {
            "query_text": "Hai nhân vật đã ở bên nhau bao lâu trước khi chia tay tại sân bay?",
            "filter_dict": {"author": "Hà Thanh Phúc"},
            "gold_answer": "Hai người ở bên nhau đúng 48 tiếng (hai ngày) — từ lúc gặp nhau tại quán cà phê trên đường Nguyễn Trãi vào thứ Bảy cho đến khi anh ra sân bay về Nhật vào gần 1 giờ sáng ngày thứ Hai.",
            "source_file": "48 giờ yêu nhau - Hà Thanh Phúc.txt"
        },
        {
            "query_text": "Vì sao Mẫn Huy bỏ nhà ra đi?",
            "filter_dict": None,
            "gold_answer": "Vì Mẫn Huy bị gia đình ép cưới Đan Uyên và ràng buộc phải cưới mới được làm giám đốc công ty. Anh không muốn sống lệ thuộc nên rời nhà, chấp nhận tay trắng để theo đuổi đam mê tranh cát.",
            "source_file": "Anh Sẽ Đến - Song Mai _ Song Châu.txt"
        },
        {
            "query_text": "Tại sao nhân vật lại quyết định vượt biển?",
            "filter_dict": None,
            "gold_answer": "Thực chất nhân vật không hề 'vượt biển' ra đại dương. Đây là ấn dụ cho việc 'vượt cạn' (sinh con). Nhân vật nữ phải vào viện sinh một bé trai (cu Vũ) trong một ca sinh khó nhưng cuối cùng đã mẹ tròn con vuông.",
            "source_file": "Anh ơi, cùng nhau ta vượt biển.... - Áo Vàng.txt"
        }
    ]
    
    top_k = 5
    json_output = {
        "strategy": f"Recursive-{chunk_size}-top{top_k}",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_config": {
            "chunk_size": chunk_size,
            "overlap": 0
        },
        "num_queries": len(queries),
        "num_chunks": num_chunks,
        "query_results": []
    }
    
    print("\n--- Thực hiện Benchmark Queries ---")
    for i, q in enumerate(queries, 1):
        query_text = q["query_text"]
        filter_dict = q["filter_dict"]
        print(f"\n[Query {i}] {query_text} (Filter: {filter_dict})")
        
        if filter_dict:
            results = store.search_with_filter(query_text, top_k=top_k, metadata_filter=filter_dict)
        else:
            results = store.search(query_text, top_k=top_k)
            
        print(f"  -> Đã trả về {len(results)} chunks")
        
        retrieved_chunks = []
        for rank, res in enumerate(results, 1):
            chunk_data = {
                "rank": rank,
                "chunk_id": res["id"],
                "score": round(res["score"], 4),
                "source": res["metadata"].get("source", ""),
                "preview": res["content"][:150].replace('\n', ' ')
            }
            retrieved_chunks.append(chunk_data)
            
        json_output["query_results"].append({
            "query_id": i,
            "query_text": query_text,
            "gold_answer": q["gold_answer"],
            "source_file": q["source_file"],
            "retrieved_chunks": retrieved_chunks
        })
        
    out_file = "my_detailed_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
        
    print(f"\nĐã xuất kết quả ra file: {out_file}")

if __name__ == "__main__":
    main()
