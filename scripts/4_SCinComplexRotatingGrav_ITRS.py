#  Код прогнозирования движения КА с учётом несферичности гравитационного
#поля Земли
#
#  В данной версии неравномерность вращения Земли учитывается
#только при определении начальных параметров
#
#           Автора: Рожков Мирослав Андреевич.  2023/07/12 
#

import plotly.graph_objects as go # Для создание интерактивного графика
import numpy as np # работа с массивами
from scipy.integrate import solve_ivp # Численное интегрирование
from scipy import special # Вычисление присоединённых функций Лежандра
# блок библиотек astropy для перехода от GCRS к ITRS
from astropy.coordinates import CartesianRepresentation, SphericalRepresentation, GCRS, ITRS  # Описание координат и систем
from astropy.time import Time # Форматирование времени
import astropy.units as u # Система единиц иземерения

# Родительский класс векторов
class Vector():
    dim = "" # единицы измерения

    def __init__(self, name, x, y, z):
        self.name = name # отображение при выводе
        self.x = x
        self.y = y
        self.z = z
        self.numcord = np.array([x, y, z]) # координаты в формате numpy

    def __str__(self):
        return f"    {self.x}\n{self.name} = {self.y}     {self.dim}\n    {self.z}"

    # матричное умножение вектора на матрицу трансформации с обновлением параметров
    def rotate(self, A):
        self.numcord = np.matmul(A, self.numcord)
        self.x, self.y, self.z = self.numcord[0], self.numcord[1], self.numcord[2]

class Position(Vector): # Координаты точки
    dim = "м"

    def __init__(self, name, x, y, z):
        super().__init__(name, x, y, z)
        self.norm = np.linalg.norm(self.numcord)

class Velocity(Vector): # Скорость
    dim = "м/с"

class Acceleration(Vector): # Ускорение
    dim = "м/с2"

# Используемые постоянные    
fm = 3.986004415E14 # грав. параметр Земли, м3/с2
R0 = 6378136.3 # экваториальный радиус Земли, м
w0 = 7.292115e-5 # угловая скорость вращения Земли, 1/с

# Переход от инерциальной GCRS к вращающейся с Землёй ITRS на базе библиотек astropy
def GCRStoITRS(time, coords):
    cart_coords = CartesianRepresentation(x=coords, copy=True)
    spher_coords = SphericalRepresentation.from_cartesian(cart_coords)
    GCRS_coords = GCRS(spher_coords, obstime=time) # данная СК в библиотеке работает тоько со сферическими координатами
    ITRS_coords = GCRS_coords.transform_to(ITRS(obstime=time))

    return np.array([ITRS_coords.x.value, ITRS_coords.y.value, ITRS_coords.z.value])

# Переход от вращающейся с Землёй ITRS к инерциальной GCRS на базе библиотек astropy
def ITRStoGCRS(time, coords):
    ITRS_coords = ITRS(x=coords, obstime=time)
    GCRS_coords = ITRS_coords.transform_to(GCRS(obstime=time)) # данная СК в библиотеке работает тоько со сферическими координатами
    cart_coords = CartesianRepresentation.from_representation(GCRS_coords)

    return np.array([cart_coords.x.value, cart_coords.y.value, cart_coords.z.value])
    
# Вычисление нормированных присоединённых функций Лежандра для модели поля Земли EGM96
def Legendre(m, n, z):
    # Готовые операторы расчёта полиномов
    Pol, dPol = special.lpmn(m, n, z)
    # Особенности оператора
    for i in range(0, m+1):
        Pol[i, :] = (-1)**i * Pol[i, :]
        dPol[i, :] = (-1)**i * dPol[i, :]

    # нормировка согласно EGM96
    for j in range(0, n+1):
           Pol[0, j] = np.sqrt(2*j+1)*Pol[0, j]
           dPol[0, j] = np.sqrt(2*j+1)*dPol[0, j]
    for i in range(1, m+1):
        for j in range(i, n+1):
            factor = np.sqrt(2*(2*j+1)*special.factorial(j-i)/special.factorial(j+i))
            Pol[i, j] = factor*Pol[i, j]
            dPol[i, j] = factor*dPol[i, j]

    return Pol, dPol

# Вычисление гравитационного ускорений согласно EGM96
def geoaccel(t, P, coeffs):
    # поворот СК в соответствие с равномерным вращением Земли
    sinwt = np.sin(w0*t)
    coswt = np.cos(w0*t)
    Earth_rot = np.array([[coswt,  sinwt, 0],
                          [-sinwt, coswt, 0],
                          [0,      0,     1]])
    P.rotate(Earth_rot)
    # определение сферических координат
    r = P.norm
    az = np.arctan2(P.y, P.x) # долгота
    cosmaz, sinmaz = np.zeros(m_max+1), np.zeros(m_max+1)
    for i in range(0, m_max+1):
        cosmaz[i] = np.cos(i*az)
        sinmaz[i] = np.sin(i*az)
    # синус и косинус широты
    sinel = P.z/r
    cosel = np.sqrt(1-sinel**2)

    g = Acceleration("g", -fm/r**2, 0, 0) # ускорение от сферичного шара

    Equator_Singularity = False # проверка на точки сингулярности
    if cosel == 1: # в полюсах
        pass
    else:
        if sinel == 0: # в экваторе
            Equator_Singularity = True

        # Расчёт по EGM96 в сферических координатах g(radius, elevation, azimuth)
        Pmn, dPmn = Legendre(m_max, n_max, sinel)
        for i in range(0, coeffs_row):
            n = int(coeffs[i, 0])
            m = int(coeffs[i, 1])
            C = coeffs[i, 2]
            S = coeffs[i, 3]
            fmR0nRmn2 = fm * R0**n / r**(n+2)

            g.x = g.x - fmR0nRmn2 * (n+1) * Pmn[m, n]*(C*cosmaz[m]+S*sinmaz[m]) # g_r
            g.y = g.y + fmR0nRmn2 * dPmn[m, n]*(C*cosmaz[m]+S*sinmaz[m])        # g_el
            if not Equator_Singularity:
                g.z = g.z + fmR0nRmn2 * m * Pmn[m, n]*(S*cosmaz[m]-C*sinmaz[m]) / sinel # g_az

        g.numcord = np.array([g.x, g.y, g.z])

    # Переход из сферических в прямоугольные координаты
    cosaz = cosmaz[1]
    sinaz = sinmaz[1]
    rot = np.array([[cosel*cosaz, sinel*cosaz, -sinaz],
                    [cosel*sinaz, sinel*sinaz,  cosaz],
                    [sinel,       -cosel,           0]])
    g.rotate(rot)

    # Возвращение в инерциальную ГСК
    g.rotate(np.transpose(Earth_rot))

    return g
    
# Правые части ОДУ
def SimpleMotion(t, y):

    P = Position("P", y[0], y[1], y[2])
    V = Velocity("V", y[3], y[4], y[5])
    fmRm3 = -fm/P.norm**3

    return [V.x, V.y, V.z, P.x*fmRm3, P.y*fmRm3, P.z*fmRm3]

def ComplexGravity(t, y):

    Pt = Position("P", y[0], y[1], y[2])
    Vt = Velocity("V", y[3], y[4], y[5])
    g = geoaccel(t, Pt, coeffs)

    return [Vt.x, Vt.y, Vt.z, g.x, g.y, g.z]
    
# Начальные данные
t0 = Time('2022-11-28 11:00:00', format='iso', scale='utc')
P0 = Position("P0", 1702631.521, 126415.744, 6769207.534)
V0 = Velocity("V0", -5734.531, -4667.074, 1528.123)
print(f"Фазовые параметры КА на момент времени {t0}")
print(P0)
print("-------------------------------")
print(V0)
coords = GCRStoITRS(t0, P0.numcord)
P0_ITRS = Position("P0(ITRS)", coords[0], coords[1], coords[2])
print(f"Положение КА на момент времени {t0} в связанной с Землёй системе координат ITRF")
print(P0_ITRS)
print("--------------------")
if P0.norm == P0_ITRS.norm:
    print(f"Перевод прошёл кооректно, геоцентрическое расстояние КА не изменилось:")
else:
    print(f"(!)Высота орбиты изменилась(!)")
print(f"Высота орбиты в GCRS = {P0.norm-R0}, м.")
print(f"Высота орбиты в ITRS = {P0_ITRS.norm-R0}, м.")

# Загрузка коэффициентов C и S из таблицы модели EGM96
coeffs_all = np.loadtxt(fname="egm96_to360.txt")
m_max, n_max = 10, 10
coeffs = coeffs_all[:63,:] # только коэффициенты до 10х10 гармоники
coeffs_row = np.size(coeffs[:,0])

# Численное интегрирование
step = 60 # шаг записи данных, с
start = 0 # начало записи данных, с
t_end = 24*3600 # конец интервала интегрирования, с
end = t_end + step
t_eval = np.arange(start, end, step)
sol = solve_ivp(ComplexGravity, [0, t_end], np.append(P0_ITRS.numcord, V0_ITRS.numcord),\
                t_eval=t_eval, rtol=1e-3, atol=0.1, max_step=30) # абсолютная погрешность - 0,1 м; макс. шаг - 30 сек.
ideal = solve_ivp(SimpleMotion, [0, t_end], np.append(P0.numcord, V0.numcord),\
                t_eval=t_eval, rtol=1e-3, atol=0.1, max_step=30)
print(sol.success)

# Запишем полученный результат в "фиксированной" ITRF
Pk = Position("Pk", sol.y[0,-1], sol.y[1,-1], sol.y[2,-1])
Vk = Velocity("Vk", sol.y[3,-1], sol.y[4,-1], sol.y[5,-1])
# Переведём этот результат обратно в GCRS с учётом прошедших суток
coords = ITRStoGCRS(t0, Pk.numcord)
Pk = Position("Pk", coords[0], coords[1], coords[2])
coords = ITRStoGCRS(t0, Vk.numcord)
Vk = Velocity("Vk", coords[0], coords[1], coords[2])
print(f"Фазовые параметры КА в GCRS на момент времени {t0+sol.t[-1]*u.s}, с")
print(Pk)
print("-------------------------")
print(Vk)
print(Pk.norm)

# Построение графиков
# Переведём каждую записанную точку результата в GCRS
Temp = sol.y[0:3,:]
N = np.size(Temp[0,:])
Pt = np.zeros((3,N))
for i in range(0, N):
    coords = ITRStoGCRS(t0, Temp[0:3,i])
    Pt[0:3,i] = coords
    
fig = go.Figure(data=[go.Scatter3d(x=Pt[0,:], y=Pt[1,:], z=Pt[2,:], name='10х10 Гармоник', mode='lines', line=dict(width=8)),
                      go.Scatter3d(x=ideal.y[0,:], y=ideal.y[1,:], z=ideal.y[2,:], name='Центральная грав.', mode='lines', line=dict(width=8, color='red'))])
B_complex = np.transpose(np.stack((Pt[:,0], Pt[:,-1])))
B_ideal = np.transpose(np.stack((ideal.y[0:3,0], ideal.y[0:3,-1])))
fig.add_traces(go.Scatter3d(x=B_complex[0,:], y=B_complex[1,:], z=B_complex[2,:], name='10x10 Гармоник', mode='markers', marker=dict(size=10)))
fig.add_traces(go.Scatter3d(x=B_ideal[0,:], y=B_ideal[1,:], z=B_ideal[2,:], name='Центральная грав.', mode='markers', marker=dict(size=8, color='orange')))
fig.update_layout(height=600, width=800)
fig.show()

R = np.linalg.norm(sol.y[0:3, :], axis=0)
R_ideal = np.linalg.norm(ideal.y[0:3, :], axis=0)
fig1 = go.Figure(data=go.Scatter(x=sol.t/3600, y=(R-R0), name='10х10 Гармоник'))
fig1.add_trace(go.Scatter(x=sol.t/3600, y=(R_ideal-R0), name='Центральная грав.'))
fig1.update_layout(height=600, width=800)
fig1.show()