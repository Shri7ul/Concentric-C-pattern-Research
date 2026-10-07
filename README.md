## Environment তৈরি

তারপর:

```bash
python -m venv .venv
```

Windows হলে:

```bash
.venv\Scripts\activate
```

তারপর:

```bash
pip install numpy matplotlib opencv-python
```

Check:

```bash
python -c "import numpy, cv2, matplotlib; print('Everything OK')"
```

যদি:

```text
Everything OK
```

আসে, ready।

---