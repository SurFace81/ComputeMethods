# -- coding: cp1251 --
import matplotlib.pyplot as plt
import numpy as np

# ������� ������ ��� ��������
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# ������� ����� ���� ��� ��������
plt.figure()

# �������� ������ ������
plt.subplot(2, 1, 1)  # (rows, columns, panel number)
plt.plot(x, y1)
plt.title('������ ������')

# �������� ������ ������
plt.subplot(2, 1, 2)
plt.plot(x, y2)
plt.title('������ ��������')

# ���������� �������
plt.show()
