import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Inisialisasi model dan tokenizer dari Hugging Face
model_name = 'gpt2'  # Anda dapat menggunakan model lain jika diperlukan
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Fungsi untuk menghasilkan teks menggunakan model generatif
def generate_text(prompt, max_length=150):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Data Dummy untuk Rencana Bisnis
business_data_list = [
    {
        'name': 'Kedai Kopi Sehat',
        'target_market': 'Konsumen muda yang peduli kesehatan',
        'products': ['kopi organik', 'snack sehat', 'minuman herbal'],
        'strategy': 'Membuka cabang di pusat perbelanjaan dan melakukan promosi di media sosial'
    },
    {
        'name': 'Toko Baju Eco-Friendly',
        'target_market': 'Kaum milenial yang peduli lingkungan',
        'products': ['kaos dari bahan daur ulang', 'tas belanja ramah lingkungan'],
        'strategy': 'Menggunakan influencer media sosial dan kampanye online'
    },
    {
        'name': 'Restoran Vegan Lezat',
        'target_market': 'Masyarakat yang mencari pilihan makanan sehat',
        'products': ['burger vegan', 'salad segar', 'smoothie bowl'],
        'strategy': 'Menyelenggarakan acara komunitas dan workshop masakan'
    },
    {
        'name': 'Pusat Kebugaran Digital',
        'target_market': 'Pekerja remote dan penggemar kebugaran',
        'products': ['program latihan online', 'kelas yoga virtual'],
        'strategy': 'Menyediakan trial gratis dan konten di YouTube'
    },
]

# Menghasilkan Rencana Bisnis untuk setiap data
for business_data in business_data_list:
    business_plan_prompt = (
        f"Rencana Bisnis untuk {business_data['name']}:\n"
        f"Target Pasar: {business_data['target_market']}\n"
        f"Produk: {', '.join(business_data['products'])}\n"
        f"Strategi Pemasaran: {business_data['strategy']}\n"
        "Rincian Rencana Bisnis:\n"
    )

    business_plan = generate_text(business_plan_prompt)
    print("Rencana Bisnis:\n", business_plan)

    # Data Dummy untuk Materi Pemasaran
    marketing_data = {
        'product_name': random.choice(business_data['products']),
        'key_features': ['100% alami', 'ramah lingkungan', 'menyehatkan'],
        'call_to_action': 'Coba sekarang dan rasakan perbedaannya!'
    }

    # Mengembangkan Materi Pemasaran
    marketing_prompt = (
        f"Deskripsi Produk: {marketing_data['product_name']}\n"
        f"Ciri Utama: {', '.join(marketing_data['key_features'])}\n"
        "Bergabunglah dengan kami dan:\n"
    )

    marketing_material = generate_text(marketing_prompt)
    print("\nMateri Pemasaran:\n", marketing_material)
    print("=" * 50)  # Pembatas untuk hasil yang lebih jelas
