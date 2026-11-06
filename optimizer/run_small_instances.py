import json
import time
import glob
from ortools.linear_solver import pywraplp

def solve_small_instances():
    # Klasördeki tüm _small.json dosyalarını bul
    json_files = glob.glob("*_small.json")
    
    if not json_files:
        print("Hiç küçük senaryo dosyası (_small.json) bulunamadı!")
        return

    print(f"Toplam {len(json_files)} adet küçük senaryo bulundu. Çözüm başlatılıyor...\n" + "-"*50)

    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            print(f"{file_path} için çözücü başlatılamadı.")
            continue

        start_time = time.time()
        
        # Test değişkeni (Eniyileme altyapısının çalıştığını doğrulamak için)
        x = solver.IntVar(0, 1, 'x')
        solver.Maximize(x)
        
        status = solver.Solve()
        end_time = time.time()
        
        solve_duration = end_time - start_time
        
        print(f"Dosya: {file_path}")
        if status == pywraplp.Solver.OPTIMAL:
            print(f"  -> Durum: Başarılı")
            print(f"  -> Amaç Değeri (Objective Value): {solver.Objective().Value()}")
            print(f"  -> Çözüm Süresi (Solve Time): {solve_duration:.4f} saniye")
        else:
            print(f"  -> Eniyileme çözüme ulaşamadı.")
        print("-" * 50)

if __name__ == '__main__':
    solve_small_instances()