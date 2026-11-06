def save_case_study_to_file(result, config, filename):
    """将案例研究结果保存到文本文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("COGNITIVE DIAGNOSIS CASE STUDY REPORT\n")
        f.write("="*80 + "\n\n")
        
        # 基本信息
        f.write("EXPERIMENT INFORMATION:\n")
        f.write("-"*40 + "\n")
        f.write(f"Method: {config['method']}\n")
        f.write(f"Dataset: {config['data_type']}\n")
        f.write(f"Student ID: {result['student_id']}\n")
        f.write(f"Analysis Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Seed: {config['seed']}\n\n")
        
        # 交互历史
        f.write("1. INTERACTION HISTORY:\n")
        f.write("-"*40 + "\n")
        f.write(f"Total interactions: {result['total_interactions']}\n")
        f.write(f"Overall Accuracy: {result['accuracy']:.4f}\n\n")
        
        f.write("Detailed Interaction Records:\n")
        for i, interaction in enumerate(result['interaction_details']):
            f.write(f"  Interaction {i+1}: Exercise {interaction['exercise_id']}, "
                   f"Score={interaction['score']}, Correct={interaction['correct']}, "
                   f"Knowledge Concepts={interaction['knowledge_concepts']}\n")
        f.write("\n")
        
        # 认知诊断结果
        f.write("2. COGNITIVE DIAGNOSIS RESULTS:\n")
        f.write("-"*40 + "\n")
        if 'student_embedding' in result:
            embedding = result['student_embedding']
            f.write(f"Student embedding shape: {embedding.shape}\n")
            f.write(f"Student embedding values (first 10 dimensions):\n")
            for j, val in enumerate(embedding[:10]):
                f.write(f"  Dimension {j}: {val:.6f}\n")
            f.write("\n")
            
            # 知识点掌握程度
            if 'knowledge_mastery' in result:
                f.write("Knowledge Mastery Levels:\n")
                for k, mastery in result['knowledge_mastery'].items():
                    f.write(f"  Knowledge {k}: {mastery:.4f}\n")
                f.write("\n")
        
        # 表现统计
        f.write("3. PERFORMANCE SUMMARY:\n")
        f.write("-"*40 + "\n")
        f.write(f"Total questions attempted: {result['total_interactions']}\n")
        f.write(f"Correct answers: {result['correct_answers']}\n")
        f.write(f"Accuracy: {result['accuracy']:.4f}\n\n")
        
        # 知识点级别分析
        f.write("4. KNOWLEDGE-LEVEL ANALYSIS:\n")
        f.write("-"*40 + "\n")
        knowledge_performance = result['knowledge_performance']
        for knowledge_id, perf in sorted(knowledge_performance.items()):
            k_accuracy = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
            f.write(f"Knowledge {knowledge_id}: {perf['correct']}/{perf['total']} "
                   f"({k_accuracy:.4f})\n")
        f.write("\n")
        
        # 预测结果
        if 'predictions' in result:
            f.write("5. PREDICTION ON SAMPLE EXERCISES:\n")
            f.write("-"*40 + "\n")
            for pred_info in result['predictions']:
                f.write(f"Exercise {pred_info['exercise_id']}: "
                       f"predicted score = {pred_info['predicted_score']:.4f}\n")
        
        # 学习建议
        f.write("6. LEARNING RECOMMENDATIONS:\n")
        f.write("-"*40 + "\n")
        if 'recommendations' in result:
            for i, rec in enumerate(result['recommendations']):
                f.write(f"{i+1}. {rec}\n")
        else:
            # 生成基本建议
            if result['accuracy'] < 0.6:
                f.write("1. Focus on fundamental concepts and review basic knowledge points.\n")
                f.write("2. Practice more exercises in weak areas to improve understanding.\n")
            elif result['accuracy'] < 0.8:
                f.write("1. Continue practicing with moderate difficulty exercises.\n")
                f.write("2. Pay attention to consistent performance across all knowledge points.\n")
            else:
                f.write("1. Excellent performance! Consider challenging yourself with advanced topics.\n")
                f.write("2. Help peers who might be struggling with concepts you've mastered.\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")


def perform_case_study(model, datahub, student_id, config):
    """执行单个学生的认知诊断案例研究"""
    print(f"\n{'='*60}")
    print(f"COGNITIVE DIAGNOSIS CASE STUDY - Student {student_id}")
    print(f"{'='*60}")
    
    # 获取该学生的所有交互记录
    all_interactions = datahub.response
    student_interactions = all_interactions[all_interactions[:, 0] == student_id]
    
    print(f"\n1. INTERACTION HISTORY:")
    print(f"{'-'*40}")
    print(f"Total interactions: {len(student_interactions)}")
    
    concept_map = datahub.get_concept_map()
    
    # 分析每道题目的交互情况
    interaction_details = []
    for i, interaction in enumerate(student_interactions):
        exercise_id = int(interaction[1])
        score = interaction[2]
        knowledge_concepts = concept_map.get(exercise_id, [])
        
        interaction_details.append({
            'interaction_id': i,
            'exercise_id': exercise_id,
            'score': score,
            'knowledge_concepts': knowledge_concepts,
            'correct': bool(score == 1)
        })
        
        print(f"  Exercise {exercise_id}: Score={score}, Correct={bool(score)}, "
              f"Knowledge Concepts={knowledge_concepts}")
    
    # 获取认知诊断结果
    print(f"\n2. COGNITIVE DIAGNOSIS RESULTS:")
    print(f"{'-'*40}")
    
    student_embedding = None
    knowledge_mastery = {}
    
    if hasattr(model, 'student_embedding'):
        # 对于有学生嵌入的模型
        student_embedding = model.student_embedding[student_id]
        if hasattr(student_embedding, 'detach'):
            student_embedding = student_embedding.detach().cpu().numpy()
        
        print(f"Student embedding shape: {student_embedding.shape}")
        print(f"Student embedding (first 10 dims): {student_embedding[:10]}")
        
        # 如果嵌入维度等于知识点数量，可以直接解释为掌握程度
        if len(student_embedding) == datahub.knowledge_num:
            print(f"\nKnowledge Mastery Levels:")
            for k in range(datahub.knowledge_num):
                mastery = student_embedding[k]
                knowledge_mastery[k] = mastery
                if k < 10:  # 只显示前10个知识点
                    print(f"  Knowledge {k}: {mastery:.4f}")
    
    # 计算学生表现统计
    total_questions = len(student_interactions)
    correct_answers = sum(1 for inter in student_interactions if inter[2] == 1)
    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    
    print(f"\n3. PERFORMANCE SUMMARY:")
    print(f"{'-'*40}")
    print(f"Total questions attempted: {total_questions}")
    print(f"Correct answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # 按知识点分析表现
    print(f"\n4. KNOWLEDGE-LEVEL ANALYSIS:")
    print(f"{'-'*40}")
    
    knowledge_performance = {}
    for interaction in interaction_details:
        for knowledge_id in interaction['knowledge_concepts']:
            if knowledge_id not in knowledge_performance:
                knowledge_performance[knowledge_id] = {'total': 0, 'correct': 0}
            
            knowledge_performance[knowledge_id]['total'] += 1
            if interaction['correct']:
                knowledge_performance[knowledge_id]['correct'] += 1
    
    for knowledge_id, perf in sorted(knowledge_performance.items()):
        k_accuracy = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
        print(f"  Knowledge {knowledge_id}: {perf['correct']}/{perf['total']} "
              f"({k_accuracy:.4f})")
    
    # 预测学生在新题目上的表现
    print(f"\n5. PREDICTION ON SAMPLE EXERCISES:")
    print(f"{'-'*40}")
    
    predictions = []
    # 选择几个代表性的题目进行预测
    sample_exercises = list(range(min(5, datahub.exercise_num)))
    for ex_id in sample_exercises:
        try:
            # 使用模型进行预测
            if hasattr(model, 'predict'):
                pred = model.predict(
                    torch.tensor([student_id], dtype=torch.int64),
                    torch.tensor([ex_id], dtype=torch.int64),
                    torch.tensor([datahub.q_matrix[ex_id]], dtype=config['dtype'])
                )
                if hasattr(pred, 'detach'):
                    pred = pred.detach().cpu().numpy()[0]
                print(f"  Exercise {ex_id}: predicted score = {pred:.4f}")
                predictions.append({
                    'exercise_id': ex_id,
                    'predicted_score': pred
                })
        except:
            pass
    
    # 生成学习建议
    recommendations = []
    if accuracy < 0.6:
        recommendations = [
            "Focus on fundamental concepts and review basic knowledge points.",
            "Practice more exercises in weak areas to improve understanding.",
            "Seek help from instructors or peers for challenging topics."
        ]
    elif accuracy < 0.8:
        recommendations = [
            "Continue practicing with moderate difficulty exercises.",
            "Pay attention to consistent performance across all knowledge points.",
            "Identify specific knowledge gaps and target them systematically."
        ]
    else:
        recommendations = [
            "Excellent performance! Consider challenging yourself with advanced topics.",
            "Help peers who might be struggling with concepts you've mastered.",
            "Explore real-world applications of the knowledge you've acquired."
        ]
    
    # 构建结果字典
    result = {
        'student_id': student_id,
        'total_interactions': total_questions,
        'correct_answers': correct_answers,
        'accuracy': accuracy,
        'interaction_details': interaction_details,
        'knowledge_performance': knowledge_performance,
        'predictions': predictions,
        'recommendations': recommendations
    }
    
    if student_embedding is not None:
        result['student_embedding'] = student_embedding
    if knowledge_mastery:
        result['knowledge_mastery'] = knowledge_mastery
    
    return result
