import React, { useState } from 'react';
import { PlayIcon } from 'lucide-react';
import { format, parseISO } from 'date-fns';
import { Chip } from "@mui/material"
import { Card, CardContent } from "@/components/ui/card";

import PlanningTab from "../tacticalPlan/page";

import NoDataFullPage from "@/components/noDataPage";

type CardItem = {
  id: string;
  name: string;
  description: string;
  start_date: string;
  end_date: string;
};

const cardData: CardItem[] = [
  {
    id: '1',
    name: 'Campaign A',
    description: 'Promo for summer sale.',
    start_date: '2025-06-01',
    end_date: '2025-06-30',
  },
  {
    id: '2',
    name: 'Campaign B',
    description: 'New product launch.',
    start_date: '2025-07-01',
    end_date: '2025-07-15',
  },
];

const formatUtcToLocal = (utcString: string): string => {
  const date = new Date(utcString + 'Z'); // Parses UTC ISO string
  return format(date, 'dd MMM yyyy hh:mm a') // Formats in local time
}

const CardList = ({ simulatorsList }) => {

    const [activeTab, setActiveTab] = useState(false);
    const [details, setDetails] = useState({});

  const onViewPlan = (obj) => {
    setDetails(obj);
    setActiveTab(true);
  };

  const onClosePlan = (id) => {
    setDetails({});
    setActiveTab(false);
  };

  return (
    <div className={activeTab ? '' : 'space-y-3 p-4'}>
      {!activeTab && (
      <h5 className="text-xl mt-0">What-If Simulators</h5>
      )}
      {!activeTab && (
      <div className="flex align-items-center gap-4">
       <div className="w-full sm:w-1/2 h-auto overflow-y-auto" style={{ maxHeight: 'calc(100vh - 200px)' }}>
          {simulatorsList && simulatorsList?.length > 0 && simulatorsList.map((card) => (
            <div
              key={card.id}
              className="relative shadow-md p-3 mb-3 h-auto executive-card transition duration-200 ease-in-out"
            >
              {/* Top-right: Play Button */}
                <button
                    className="absolute right-4 top-1/2 -translate-y-1/2 z-10 p-2 rounded-full bg-white hover:bg-gray-100 dark:bg-gray-800 dark:hover:bg-gray-700 shadow"
                    title="Play"
                    onClick={() => onViewPlan(card)}
                    >
                    <PlayIcon className="w-6 h-6 text-blue-600" />
                </button>

              {/* Title & Description */}
                <div className="space-y-2 pr-16">
                  <p className="text-sm m-0 text-gray-600 dark:text-gray-200 flex items-center" style={{ opacity: 0.5 }}>{card.reference} -  <Chip label={card.state} size="small" className="ml-2 rounded-0" color={card.state === 'Draft' ? 'primary' : card.state === 'Approved' ? 'success' : 'error'} /></p>
                    <h3 className="text-lg mt-0 font-semibold text-gray-900 dark:text-white" style={{ lineHeight: '15px' }}>{card.name}</h3>
                </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-3 xl:grid-cols-4 gap-2 mt-2">
                    <div className="">
                      <p className="text-xs text-gray-600 dark:text-gray-200" style={{ opacity: 0.5 }}>Created by</p>
                      <p className="text-sm mb-0 text-gray-600 dark:text-gray-200">{card.create_uid && card.create_uid?.[1] || ''}</p>
                    </div>
                    <div className="">
                      <p className="text-xs text-gray-600 dark:text-gray-200" style={{ opacity: 0.5 }}>Created on</p>
                      <p className="text-sm mb-0 text-gray-600 dark:text-gray-200">{card.create_date ? formatUtcToLocal(card.create_date) : ''}</p>
                    </div>
                    <div className="">
                     <p className="text-xs text-gray-600 dark:text-gray-200" style={{ opacity: 0.5 }}>Last Updated on</p>
                     <p className="text-sm mb-0 text-gray-600 dark:text-gray-200">{card.write_date ? formatUtcToLocal(card.write_date) : ''}</p>
                     </div>
                  </div>
            </div>
          ))}
      </div>
      <div className="w-full sm:w-1/2 h-auto">
        <Card className="p-5 w-full h-auto transition duration-200 ease-in-out" style={{ borderLeft: '3px solid grey' }}>
          <h4 className="text-xl">What-if Scenarios</h4>
          <p className="text-lg mt-2 mb-2">
            A What-If Simulator for supply chain planning allows businesses to model scenarios, stress-test strategies, and optimize decisions before execution.
          </p>
           <p className="text-lg mt-2 mb-2">
            Supply chains are complex networks vulnerable to disruptions—demand fluctuations, supplier delays, geopolitical risks, and natural disasters.
          </p>
          <p>Features: </p>
          <div className="p-2">
            <ul>
              <li className="text">1) Test parallel strategies</li>
              <li className="text">2) Share simulations with stakeholders</li>
              <li className="text">3) Track decision logic</li>
              <li className="text-">4) Apply changes to live systems safely</li>
            </ul>
          </div>
        </Card>
      </div>
      </div>
      )}
      {activeTab && (
        <PlanningTab whatIfDetails={details} navigateSimulator={onClosePlan} businessId={0} />
      )}
      {!(simulatorsList && simulatorsList?.length > 0) && (
        <NoDataFullPage message="No Stimulators are available." />
      )}
    </div>
  );
};

export default CardList;

