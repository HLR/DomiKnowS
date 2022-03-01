# Sensor class

`Sensor`s are procedures to access external resources and procedures. For example, reading from raw data, feature engineering staffs, and preprocessing procedures.
"Sensors" are looked at as blackboxes in a program. Users use sensors as callable objects. Underlying `forward()` function will be used to calculate an output. One can override `forward()` function to customize how the sensor get the output.
Base classes of `Sensor` that in retuen would handle details like caching calcuated results, managing invocation path, and converting input.


```python
class Sensor(BaseGraphTreeNode):
    def __call__(
        self,
        data_item: Dict[str, Any],
        force=False,
        sensor_name="None"
    )
```

All sensors are called during training. The call function for the Base Sensor is simple it just tries to evoke the `update_context`. It `update_context` is executed successfully, its output is return. Otherwise, an Error is retured specifying the name of the sensor.
`__call__` is immidiatly overridden by the `TorchSensor` and the only difference would be to save the `data_item` vaiable in the Class attributes (`self.context_helper`). The only other Sensor that overrides `__call__` is `JointSensor`.
`data_item` is a dictionary that hold all the properties that are calculated by the sensors. At the beggining of the program `data_item` is the dictionary read from the reader provided during training.

```python
def update_context(
    self,
    data_item: Dict[str, Any],
    force=False
)
```
`update_context` calls the forward function to use the sensor to produce the new feature. It also handles the preperation and fethicng of inputs to the forward function.

```python
def forward(
    self,
    data_item: Dict[str, Any]
)
```
The `forward` function uses the input data to produce the new feature and add it to the `data_item` which is the main functioanlity of the Sensors.

```python
@property
def prop(self):
    if self.sup is None:
        raise ValueError('{} must be used with with property assignment.'.format(type(self)))
    return self.sup

@property
def concept(self):
    if self.prop.sup is None:
        raise ValueError('{} must be used with with concept[property] assignment.'.format(type(self)))
    return self.prop.sup
```

Finally, Sensors have two properties defined here. When we are defining a Sensor in our main code we address it to a `Property` and a `Concept`. In this formatL
```
Concept[Property] = Sensor(*args)
```
Attribute prop helps to access the `Property` and attribute concept helps to access `Concept` in the above format.
    
