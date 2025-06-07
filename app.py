<!DOCTYPE html>
<html lang="uk">
<head>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <meta charset="UTF-8" />
    <title>Генерація і сортування зображень</title>
</head>
<body>
    <h1>Генерація і сортування зображень</h1>

    <!-- Форма для генерації -->
    <form id="generateForm">
        <label>Запит 1: <input type="text" name="target1" required /></label>
        <label>Запит 2: <input type="text" name="target2" required /></label>
        <label>Кількість: <input type="number" name="quantity" min="1" max="20" value="5" required /></label>
        <button type="submit">Згенерувати</button>
    </form>

    <!-- Контейнер з зображеннями -->
    <div class="image-container" id="imagesContainer">
        {% for image in images %}
        <div class="image-box">
            <img src="{{ url_for('static', filename='generated_images/' + image) }}" alt="{{ image }}" />
            <label><input type="checkbox" name="selected_images" value="{{ image }}" /> Обрати</label>
        </div>
        {% endfor %}
    </div>

    <br/>
    <button id="sortBtn">Сортувати вибрані</button>

<!-- Лічильник і кнопка навчання -->
<div id="statusSection" style="margin-top: 15px; text-align: center;">
    <p>Зображень для донавчання: <span id="totalCount">0</span>\50</p>
    <button id="trainBtn" disabled>Донавчити модель</button>
</div>


<script>
    async function updateImageCounts() {
        try {
            const res = await fetch('/get_counts');
            const data = await res.json();
            const total = (data.positive || 0) + (data.negative || 0);

            document.getElementById('totalCount').textContent = total;
            document.getElementById('trainBtn').disabled = total < 10; // Мінімум 10 загалом
        } catch (err) {
            console.error('Помилка підрахунку зображень:', err);
        }
    }

    document.getElementById('generateForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(e.target);
        const data = {
            target1: formData.get('target1'),
            target2: formData.get('target2'),
            quantity: formData.get('quantity'),
        };

        const response = await fetch('/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });

        if (response.ok) {
            alert('Зображення згенеровано!');
            location.reload();
        } else {
            alert('Помилка генерації.');
        }
    });

    document.getElementById('sortBtn').addEventListener('click', async () => {
        const checkedBoxes = document.querySelectorAll('input[name="selected_images"]:checked');
        const selected = Array.from(checkedBoxes).map(cb => cb.value);
        if(selected.length === 0) {
            alert('Оберіть хоча б одне зображення для сортування!');
            return;
        }

        const data = {
            selected_images: selected,
            folder_path: 'Positive' // або зроби вибір через select
        };

        const response = await fetch('/sort', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });

        const result = await response.json();
        if(response.ok) {
            alert(result.message);
            location.reload();
        } else {
            alert(result.error || 'Помилка сортування.');
        }
    });

    document.getElementById('trainBtn').addEventListener('click', async () => {
        const response = await fetch('/train', { method: 'POST' });
        const result = await response.json();

        if (response.ok) {
            alert(result.message || 'Навчання завершено');
        } else {
            alert(result.error || 'Помилка навчання');
        }
    });

    window.addEventListener('load', updateImageCounts);
</script>
</body>
</html>