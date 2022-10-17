
# Datasets
# The following 3 classes are used to handle the dataset that we have.


def preprocess(df):
    return (
        df.drop(
            ['Throttle_position_signal', 'Short_Term_Fuel_Trim_Bank1', 'Filtered_Accelerator_Pedal_value', 'Absolute_throttle_position', 'Engine_soacking_time','Inhibition_of_engine_fuel_cut_off', 'Engine_in_fuel_cut_off',
       'Fuel_Pressure', 'Engine_speed', 'Engine_torque_after_correction', 'Flywheel_torque_(after_torque_interventions)', 'Current_spark_timing', 'Engine_Idel_Target_Speed', 'Minimum_indicated_engine_torque', 'Flywheel_torque',
       'Torque_scaling_factor(standardization)', 'Standard_Torque_Ratio', 'Requested_spark_retard_angle_from_TCU', 'TCU_requests_engine_torque_limit_(ETL)', 'TCU_requested_engine_RPM_increase', 'Target_engine_speed_used_in_lock-up_module', 'Glow_plug_control_request', 'Current_Gear',
       'Engine_coolant_temperature.1', 'Wheel_velocity_rear_right-hand', 'Torque_converter_turbine_speed_-_Unfiltered', 'Clutch_operation_acknowledge', 'Converter_clutch', 'Gear_Selection', 'Vehicle_speed', 'Acceleration_speed_-_Longitudinal', 'Indication_of_brake_switch_ON/OFF', 'Master_cylinder_pressure',
       'Calculated_road_gradient', 'Acceleration_speed_-_Lateral', 'Steering_wheel_speed', 'Steering_wheel_angle', 'PathOrder'], axis=1
        )
    )


class Dataset(torch.utils.data.Dataset):
    """
    This class is used to handle the train dataset to work witht the Triplet Loss.
    The __getitem__ method (to be used with a dataloader) returns the anchor, a random postive and a random negative for that anchor
    as well as the anchor's label.
    """

    def __init__(self, data, labels, input_length):
        self.data, self.labels = data, labels
        self.index = [i for i in range(len(self.data))]
        self.length = input_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X_anchor = self.data[index]
        y_anchor = self.labels[index]

        anchor_wvlt = reference_transform(X_anchor)

        # concatenate the data for the TCN and the haar wavelet transform
        # they will be split in the forward pass
        return torch.cat((X_anchor, anchor_wvlt), 1), \
            y_anchor

    @torch.no_grad()
    def get_classifier_data(self, labels_list, model: nn.Module):
        data = []
        labels = []
        device = torch.device("cuda:0")

        index_list = [i for i in self.index if self.labels[i] in labels_list]

        for i in index_list:
            anchor = self[i][0].unsqueeze(0)
            anchor = anchor.to(device)
            embed = model(anchor)

            data.append(embed.cpu().detach().numpy().squeeze())
            labels.append(labels_list.index(self.labels[i]))

        data = np.array(data)
        return data, labels


def loaded_dataset():
    df = pd.read_csv("security dataset path file")
    df = preprocess(df)
    classes=['A','B','C','D','E','F','G','H','I','J']
    drivers =[]
    for c in classes:
        drivers.append(df[df['Class']==c])
    dataa=[]
    for c in range(len(drivers)):
        nt=0
        nv=0
        drivers[c]=drivers[c].reset_index(drop=True)
        idxs=drivers[c][drivers[c]['Time(s)']==1].index.values
    
        for i in range(len(idxs)):
            if i <(len(idxs)-1):
              nt=nt+1
              dataa.append(drivers[c][idxs[i]:idxs[i+1]])
            if i==(len(idxs)-1):
               nv=nv+1
               dataa.append(drivers[c][idxs[i]:])
            #print("Driver : "+str(c)+" number of trips :"+str(len(idxs))+ "  For Train : "+str(nt)+"  For valid :"+str(nv))
    drivers =[]

    ss=0
    for i in range(len(dataa)):
        #print(n)
        n=int(len(dataa[i])/60)
        #print(" Drive "+str(i)+" contains "+str(n)+" subdriversets")
        dd=0
        for j in range(n):
           #print(j)
            temp=dataa[i][dd:dd+60]
            temp=temp.reset_index(drop=True)
            drivers.append(temp)
            ss=ss+1
            dd=dd+60
           #print("total is "+str(ss))  
    samples = list()
    labels=list()
    for c in drivers:
        labels.append(c['Class'][0])
        del c['Class']
        del c['Time(s)']
        samples.append(c.values)
    x=[]
    for _ in samples:
        _ = _.transpose()
        x.append(torch.from_numpy(_).float())
        #x=np.array(x)
      
    return x, labels

def split_train_test(raw_data, raw_labels, ratio=0.8):
    le = preprocessing.LabelEncoder()
    le.fit(raw_labels)
    labels=le.transform(raw_labels) 
    X_train, X_test, y_train, y_test = train_test_split(raw_data, labels, train_size=0.8, random_state=100)

    return X_train, y_train, X_test, y_test




