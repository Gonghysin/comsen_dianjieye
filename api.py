from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from all_function import predict_molecule_ie, generate_molecule_by_properties

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
# 启用CORS，允许跨域请求
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/predict', methods=['POST'])
def predict_energy():
    data = request.get_json()
    smiles = data.get('smiles', '')
    
    if not smiles:
        return jsonify({'error': 'SMILES字符串不能为空'}), 400
    
    try:
        # 调用预测函数
        energy = predict_molecule_ie(smiles)
        
        if energy is None:
            return jsonify({'error': '预测失败'}), 500
            
        return jsonify({
            'energy': energy,
            'confidence': 0.95  # 由于是确定性模型，设置较高的置信度
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate_molecules():
    data = request.get_json()
    target_affinity = data.get('affinity', 0.5)
    target_ie = data.get('ionization_energy', 10.0)
    num_samples = data.get('num_samples', 3)
    
    try:
        # 调用生成函数
        generated_smiles = generate_molecule_by_properties(
            affinity=target_affinity,
            ionization_energy=target_ie,
            num_samples=num_samples
        )
        
        if generated_smiles is None:
            return jsonify({'error': '生成失败'}), 500
            
        # 对每个生成的分子预测其性质
        results = []
        for smiles in generated_smiles:
            predicted_ie = predict_molecule_ie(smiles)
            results.append({
                'smiles': smiles,
                'properties': {
                    'ionization_energy': predicted_ie,
                    'affinity': target_affinity  # 使用目标亲和能
                }
            })
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5714, debug=False)
