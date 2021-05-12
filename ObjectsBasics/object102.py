class Vehicle:
    color = "white"
    def __init__(self, name, max_speed, mileage, capacity=50):
        self.name = name
        self.max_speed = max_speed
        self.mileage = mileage
        self.capacity = capacity
        
    def seating_capacity(self):
        return "The capacity of a {} is {} passengers".format(self.name, self.capacity)

    def fare(self, rate=100):
        return self.capacity * rate

class Bus(Vehicle):
    def seating_capacity(self):
        return super().seating_capacity()

    def fare(self,maintenance=10):
        return super().fare() * (100+maintenance)/100
    pass

if __name__ == "__main__":
    bmw = Vehicle("BMW S6", 100, 20)
    print("Vehicle Name:", bmw.name, "Color:", bmw.color, "Speed:", bmw.max_speed, "Mileage:", bmw.mileage)
    print("Total Fare:", bmw.fare())

    School_bus = Bus("Volvo", 200, 30)
    print("Vehicle Name:", School_bus.name, "Color:", School_bus.color, "Speed:", School_bus.max_speed, "Mileage:", School_bus.mileage)
    print(School_bus.seating_capacity())
    print("Total Fare:", School_bus.fare())
    print("Object Type:", type(School_bus))
    print(isinstance(School_bus, Vehicle))
