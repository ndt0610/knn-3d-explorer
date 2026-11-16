import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def classify_knn(data1, data2, target, k):
    """Classify by the k closest 3D points; radius is display-only."""
    points = []
    for label, data in ((1, data1), (2, data2)):
        for point in data:
            points.append({"raw_type": label, "coords": np.asarray(point),
                           "dist": float(np.linalg.norm(np.asarray(point) - target))})
    if not points or not 1 <= k <= len(points):
        raise ValueError("k must be between 1 and the total number of points")
    points.sort(key=lambda item: item["dist"])
    neighbors = points[:k]
    votes = {label: sum(item["raw_type"] == label for item in neighbors) for label in (1, 2)}
    if votes[1] == votes[2]:
        # In a voting tie, use the label of the nearest selected point.
        winner = neighbors[0]["raw_type"]
    else:
        winner = max(votes, key=votes.get)
    return winner, votes, points

class KNN3DManualApp:
    def __init__(self, root):
        self.root = root
        self.root.title("k-NN 3D — Phân loại và giải thích")
        self.root.geometry("1420x760")

        # --- CHIA GIAO DIỆN THÀNH 2 CỘT ---
        # Cột trái: Nhập liệu và Bảng kết quả
        left_frame = ttk.Frame(root, padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Cột phải: Biểu đồ 3D
        right_frame = ttk.Frame(root, padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # === 1. PHẦN NHẬP LIỆU (Cột trái) ===
        input_group = ttk.LabelFrame(left_frame, text="Nhập tọa độ (Mỗi dòng là x,y,z)")
        input_group.pack(fill=tk.X, pady=5)

        # Nhập Loại 1
        ttk.Label(input_group, text="Dữ liệu Loại 1 (Đỏ):").grid(row=0, column=0, sticky="w")
        self.text_data1 = tk.Text(input_group, height=5, width=30)
        self.text_data1.grid(row=1, column=0, padx=5, pady=5)
        self.text_data1.insert(tk.END, "1, 2, 3\n2, 3, 4\n1.5, 2.5, 3.5") # Dữ liệu mẫu

        # Nhập Loại 2
        ttk.Label(input_group, text="Dữ liệu Loại 2 (Xanh):").grid(row=0, column=1, sticky="w")
        self.text_data2 = tk.Text(input_group, height=5, width=30)
        self.text_data2.grid(row=1, column=1, padx=5, pady=5)
        self.text_data2.insert(tk.END, "6, 7, 8\n7, 8, 9\n6.5, 7.5, 8.5") # Dữ liệu mẫu

        # Nhập Điểm cần xét & Bán kính
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=10)

        ttk.Label(control_frame, text="Điểm xét (x,y,z):").pack(side=tk.LEFT)
        self.entry_target = ttk.Entry(control_frame, width=15)
        self.entry_target.insert(0, "2.5, 3.5, 4.5")
        self.entry_target.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Bán kính (R):").pack(side=tk.LEFT, padx=10)
        self.scale_radius = tk.Scale(control_frame, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL)
        self.scale_radius.set(3.0)
        self.scale_radius.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(control_frame, text="K láng giềng:").pack(side=tk.LEFT, padx=(12, 3))
        self.spin_k = ttk.Spinbox(control_frame, from_=1, to=99, width=4)
        self.spin_k.set(3)
        self.spin_k.pack(side=tk.LEFT)

        btn_run = ttk.Button(left_frame, text="Phân loại điểm và cập nhật biểu đồ", command=self.process_data)
        btn_run.pack(fill=tk.X, pady=5)

        # === 2. BẢNG KẾT QUẢ (Cột trái - Bên dưới) ===
        ttk.Label(left_frame, text="Bảng xếp hạng khoảng cách (Gần -> Xa):").pack(pady=5, anchor="w")

        columns = ("rank", "type", "coords", "dist", "status")
        table_frame = ttk.Frame(left_frame)
        table_frame.pack(fill=tk.X, pady=2)
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=6)
        table_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=table_scrollbar.set)

        self.tree.heading("rank", text="#")
        self.tree.column("rank", width=30, anchor="center")
        self.tree.heading("type", text="Loại")
        self.tree.column("type", width=80)
        self.tree.heading("coords", text="Tọa độ")
        self.tree.column("coords", width=100)
        self.tree.heading("dist", text="Khoảng cách")
        self.tree.column("dist", width=80)
        self.tree.heading("status", text="Trạng thái")
        self.tree.column("status", width=80)

        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        table_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Nhãn kết luận
        self.lbl_result = ttk.Label(left_frame, text="Dự đoán: ...", font=("Arial", 12, "bold"), foreground="blue")
        self.lbl_result.pack(pady=10)

        # === 3. BIỂU ĐỒ 3D (Cột phải) ===
        self.fig = plt.Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def parse_coordinates(self, text_input):
        """Reject malformed rows instead of silently dropping training points."""
        points = []
        for line_number, line in enumerate(text_input.splitlines(), start=1):
            if not line.strip():
                continue
            parts = [float(x) for x in line.replace(',', ' ').split()]
            if len(parts) != 3 or not np.isfinite(parts).all():
                raise ValueError(f"Dòng {line_number} cần đúng 3 số hữu hạn")
            points.append(parts)
        return np.array(points)

    def draw_sphere(self, center, radius):
        """Vẽ khối cầu mờ"""
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        # Mặt cầu mờ + lưới để dễ thấy ranh giới
        self.ax.plot_surface(x, y, z, color='cyan', alpha=0.12, edgecolor='none')
        self.ax.plot_wireframe(x, y, z, color='gray', linewidth=0.3, alpha=0.4)

    def process_data(self):
        # 1. Lấy dữ liệu
        try:
            data1 = self.parse_coordinates(self.text_data1.get("1.0", tk.END))
            data2 = self.parse_coordinates(self.text_data2.get("1.0", tk.END))

            target_raw = [float(x) for x in self.entry_target.get().replace(',', ' ').split()]
            if len(target_raw) != 3 or not np.isfinite(target_raw).all():
                raise ValueError("Điểm xét cần đúng 3 số hữu hạn")
            target = np.array(target_raw)

            radius = self.scale_radius.get()
            k = int(self.spin_k.get())
        except ValueError as exc:
            messagebox.showerror("Dữ liệu không hợp lệ", str(exc))
            return

        if len(data1) == 0 or len(data2) == 0:
            messagebox.showwarning("Thiếu dữ liệu", "Vui lòng nhập ít nhất 1 điểm cho mỗi loại.")
            return

        # 2. Tính toán khoảng cách và tạo danh sách tổng hợp
        all_points = []

        # Xử lý Loại 1
        for i, p in enumerate(data1):
            dist = np.linalg.norm(p - target)
            status = "TRONG" if dist <= radius else "NGOÀI"
            all_points.append({
                "type": "Loại 1 (Đỏ)",
                "raw_type": 1,
                "coords": p,
                "dist": dist,
                "status": status,
                "id": i+1
            })

        # Xử lý Loại 2
        for i, p in enumerate(data2):
            dist = np.linalg.norm(p - target)
            status = "TRONG" if dist <= radius else "NGOÀI"
            all_points.append({
                "type": "Loại 2 (Xanh)",
                "raw_type": 2,
                "coords": p,
                "dist": dist,
                "status": status,
                "id": i+1
            })

        # Radius is visual context. The prediction itself uses the k nearest points.
        try:
            winner, votes, _ = classify_knn(data1, data2, target, k)
        except ValueError as exc:
            messagebox.showerror("K không hợp lệ", str(exc))
            return
        all_points.sort(key=lambda x: x["dist"])

        # 4. Hiển thị lên bảng (Treeview)
        for item in self.tree.get_children():
            self.tree.delete(item)

        count_type1_inside = 0
        count_type2_inside = 0

        for idx, item in enumerate(all_points):
            coords_str = f"({item['coords'][0]:.1f}, {item['coords'][1]:.1f}, {item['coords'][2]:.1f})"
            dist_str = f"{item['dist']:.2f}"

            # Đếm số lượng trong vòng tròn
            if item["status"] == "TRONG":
                if item["raw_type"] == 1: count_type1_inside += 1
                else: count_type2_inside += 1

                # Highlight dòng trong bảng (tag 'inside')
                self.tree.insert("", "end", values=(idx+1, item["type"], coords_str, dist_str, item["status"]), tags=('inside',))
            else:
                self.tree.insert("", "end", values=(idx+1, item["type"], coords_str, dist_str, item["status"]))

        # Tô màu cho các dòng trong bảng
        self.tree.tag_configure('inside', background='#ccffcc') # Màu xanh lá nhạt cho điểm nằm trong

        # 5. Kết luận
        res = (f"k-NN: Loại {winner} ({votes[1]}:{votes[2]} phiếu trong {k} điểm gần nhất). "
               f"Trong bán kính R: {count_type1_inside}:{count_type2_inside}.")
        self.lbl_result.config(text=res)

        # 6. Vẽ đồ thị 3D
        self.ax.clear()

        # Gom nhóm để tô khác biệt trong/ngoài
        type1_inside = np.array([item["coords"] for item in all_points if item["raw_type"] == 1 and item["status"] == "TRONG"])
        type1_outside = np.array([item["coords"] for item in all_points if item["raw_type"] == 1 and item["status"] == "NGOÀI"])
        type2_inside = np.array([item["coords"] for item in all_points if item["raw_type"] == 2 and item["status"] == "TRONG"])
        type2_outside = np.array([item["coords"] for item in all_points if item["raw_type"] == 2 and item["status"] == "NGOÀI"])

        # Vẽ điểm loại 1
        if type1_outside.size:
            self.ax.scatter(type1_outside[:,0], type1_outside[:,1], type1_outside[:,2],
                            c='none', edgecolors='red', marker='o', s=70, linewidths=1.5, label='Loại 1 - Ngoài')
        if type1_inside.size:
            self.ax.scatter(type1_inside[:,0], type1_inside[:,1], type1_inside[:,2],
                            c='red', marker='o', s=80, edgecolors='black', linewidths=0.5, label='Loại 1 - Trong')

        # Vẽ điểm loại 2
        if type2_outside.size:
            self.ax.scatter(type2_outside[:,0], type2_outside[:,1], type2_outside[:,2],
                            c='none', edgecolors='blue', marker='^', s=70, linewidths=1.5, label='Loại 2 - Ngoài')
        if type2_inside.size:
            self.ax.scatter(type2_inside[:,0], type2_inside[:,1], type2_inside[:,2],
                            c='blue', marker='^', s=80, edgecolors='black', linewidths=0.5, label='Loại 2 - Trong')

        # Vẽ điểm target
        self.ax.scatter(target[0], target[1], target[2], c='yellow', marker='*', s=300, edgecolors='black', label='Điểm xét')

        # Vẽ khối cầu
        self.draw_sphere(target, radius)

        # Vẽ đường nối từ tâm đến các điểm nằm TRONG khối cầu (để dễ nhìn)
        for item in all_points:
            if item["status"] == "TRONG":
                p = item["coords"]
                self.ax.plot([target[0], p[0]], [target[1], p[1]], [target[2], p[2]], 'k--', alpha=0.35)

        # Căn chỉnh trục tự động cho đẹp
        all_coords = np.vstack((data1, data2, target))
        max_range = np.ptp(all_coords, axis=0).max() / 2.0
        mid_x = (np.max(all_coords[:,0]) + np.min(all_coords[:,0])) * 0.5
        mid_y = (np.max(all_coords[:,1]) + np.min(all_coords[:,1])) * 0.5
        mid_z = (np.max(all_coords[:,2]) + np.min(all_coords[:,2])) * 0.5

        self.ax.set_xlim(mid_x - max_range - 1, mid_x + max_range + 1)
        self.ax.set_ylim(mid_y - max_range - 1, mid_y + max_range + 1)
        self.ax.set_zlim(mid_z - max_range - 1, mid_z + max_range + 1)

        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        self.ax.legend(loc='upper left')
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = KNN3DManualApp(root)
    app.process_data()
    root.mainloop()
