import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import * as Select from '@radix-ui/react-select';
import { Check, ChevronDown } from 'lucide-react';

type Feature = {
  title: string;
  description?: string;
  icon?: React.ReactNode;
};

type FeatureModalProps = {
    open: boolean;
    feature: Feature | null;
    onClose: () => void;
    options: any[];
    setActiveTab: (tabName: string) => void;
    setBusinessData: (opt: any) => void;
};

type Option = {
  value: string;
  label: string;
};

export const UnitSelection: React.FC<FeatureModalProps> = ({ open, options, feature, businessData, setBusinessData, setActiveTab, onClose }) => {
  const [selectedOption, setSelectedOption] = useState<string>('');
  const navigate = useNavigate();

  useEffect(() => {
    if (businessData?.id) {
      setSelectedOption(businessData.id);
    }
  }, [businessData]);

  const handleContinue = () => {
    if (selectedOption && feature) {
        setBusinessData(options.find(item => item.id === selectedOption) || (options && options.length > 0 ? options[0] : {}))
        // navigate('/dashboard');
        console.log(feature?.title);
        if(feature?.title === 'Planning'){
          setActiveTab("businessPerformance2");
        }else{
        setActiveTab("businessPerformance");
        }
        onClose();
    }
  };

  if (!open || !feature) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-md p-6 bg-white dark:bg-gray-900 rounded-lg shadow-lg">
        <h6 className="text-lg font-semibold mb-4 text-center">Choose a Business Unit</h6>

        <Select.Root value={selectedOption} onValueChange={setSelectedOption}>
          <Select.Trigger className="w-full px-4 py-2 border rounded flex items-center justify-between text-left">
            <Select.Value placeholder="Select an option..." />
            <ChevronDown className="w-4 h-4 text-muted-foreground" />
          </Select.Trigger>

          <Select.Portal>
           <Select.Content className="z-50 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-md mt-1 overflow-hidden">
                <Select.Viewport>
                    {options.map((opt) => (
                    <Select.Item
                        key={opt.id}
                        value={opt.id}
                        className="px-4 py-2 flex items-center justify-between cursor-pointer text-gray-900 dark:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700"
                    >
                        <Select.ItemText>
                           <p className="font-medium">{opt.display_name}</p>
                            <p className="text-sm text-gray-500 dark:text-gray-400">
                              {opt.description}
                            </p>
                        </Select.ItemText>
                        <Select.ItemIndicator>
                        <Check className="w-4 h-4 text-green-500" />
                        </Select.ItemIndicator>
                    </Select.Item>
                    ))}
                </Select.Viewport>
                </Select.Content>
          </Select.Portal>
        </Select.Root>

        <div className="mt-6 flex justify-end space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 border rounded text-sm hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            Cancel
          </button>
          <button
            onClick={handleContinue}
            disabled={!selectedOption}
            className="px-4 py-2 bg-blue-600 text-white text-sm rounded disabled:opacity-50"
          >
            Continue
          </button>
        </div>
      </div>
    </div>
  );
};
