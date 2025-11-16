export type WeeklyData = {
  Date: Date;
  Value: number; // Target Column (Value)
  Orders: number; // Exogenous Column (Orders)
  Forecast?: number; // Forecast Data (if available)
  ForecastLower?: number; // Lower bound of forecast
  ForecastUpper?: number; // Upper bound of forecast
  CreatedDate: Date; // Created Date
};

export type DataQuality = {
  seasonality: string;
  trend: string;
};

export type OutlierData = {
  index: number;
  value: number;
  date: Date;
  reason: string;
  severity: 'high' | 'medium' | 'low';
};

export type ForecastData = {
  date: Date;
  forecast: number;
  lower: number;
  upper: number;
  confidence: number;
};

export type ForecastMetrics = {
  modelName: string; // e.g., "XGBoost", "Prophet", "LSTM"
  accuracy: number; // e.g., 94.2
  mape: number; // Mean Absolute Percentage Error
  rmse: number; // Root Mean Squared Error
  r2: number; // R-squared score
  forecastHorizon: number; // Days forecasted
  trainedDate: Date;
  confidenceLevel: number; // e.g., 95
  isOptimal: boolean; // True if this is the best model
  comparisonMetrics?: {
    otherModels: Array<{
      name: string;
      accuracy: number;
      mape: number;
    }>;
  };
};

export type LineOfBusiness = {
  id: string;
  name: string;
  description: string;
  code: string; // Required field
  businessUnitId: string; // Reference to parent BU
  startDate: Date; // Required field
  hasData: boolean;
  dataUploaded: Date | null;
  recordCount: number;
  timeSeriesData?: WeeklyData[]; // Includes historical + forecast data
  dataQuality?: DataQuality;
  forecastMetrics?: ForecastMetrics; // Model performance after forecasting
  file?: File;
  suggestions?: string[];
  createdDate: Date; // When LOB was created
  updatedDate: Date; // Last modified
  status: 'active' | 'inactive' | 'archived';
};

export type BusinessUnit = {
  id: string;
  name: string;
  description: string;
  code: string; // Required field
  startDate: Date; // Required field
  displayName: string; // Required field
  color: string;
  lobs: LineOfBusiness[];
  createdDate: Date; // When BU was created
  updatedDate: Date; // Last modified
  status: 'active' | 'inactive' | 'archived';
};

export type AgentStatus = 'active' | 'idle' | 'error' | 'completed';

export interface DynamicInsight {
  id: string;
  title: string;
  description: string;
  value: string;
  trend?: 'up' | 'down' | 'stable';
  significance: 'high' | 'medium' | 'low';
  category: 'forecast' | 'trend' | 'seasonality' | 'anomaly' | 'quality' | 'setup' | 'data' | 'analysis' | 'session';
  actionable: boolean;
  recommendation?: string;
}

export interface DateRange {
  start: Date;
  end: Date;
  preset?: 'last_7_days' | 'last_30_days' | 'last_90_days' | 'last_year';
}

export type Agent = {
  id: string;
  name: string;
  task: string;
  status: AgentStatus;
  successRate: number;
  avgCompletionTime: number; // in ms
  errorCount: number;
  cpuUsage: number;
  memoryUsage: number;
};

export type AgentCommunication = {
  timestamp: string;
  from: string;
  to: string;
  message: string;
  type: 'task_handoff' | 'data_ready' | 'user_control' | 'system_update';
};

export type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  isTyping?: boolean;
  suggestions?: string[];
  visualization?: {
    data: WeeklyData[];
    target: 'Value' | 'Orders';
    isShowing?: boolean;
    showOutliers?: boolean; // Only show outliers for exploration/preprocessing
  };
  agentType?: string;
  canGenerateReport?: boolean;
  reportData?: any;
  showCapacityPlanning?: boolean;
  tokenUsage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    fromCache?: boolean;
  };
};

export type WorkflowStatus = 'completed' | 'active' | 'pending' | 'error';

export type WorkflowStep = {
  id: string;
  name: string;
  status: WorkflowStatus;
  dependencies: string[];
  estimatedTime: string; // for example, "2m 15s"
  details: string;
  agent?: string;
};

export type AgentMonitorProps = {
  isOpen: boolean;
};

// Form data types for BU/LOB creation
export type BUCreationData = {
  name: string;
  description: string;
  code: string;
  startDate: Date;
  displayName: string;
};

export type LOBCreationData = {
  name: string;
  description: string;
  code: string;
  businessUnitId: string;
  startDate: Date;
};

// Data validation types
export type ValidationError = {
  field: string;
  message: string;
  severity: 'critical' | 'error' | 'warning';
  row?: number;
  column?: string;
  suggestedFix?: string;
};

export type ValidationResult = {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationError[];
  suggestions: string[];
  dataPreview?: any[];
  columnMapping?: ColumnMapping;
};

export type ColumnMapping = {
  detected: {
    date?: string;
    target?: string;
    exogenous?: string;
    forecast?: string;
  };
  required: string[];
  optional: string[];
  suggestions: {
    [key: string]: string[];
  };
};

// Professional agent response types
export type ProfessionalResponse = {
  content: string;
  tone: 'informative' | 'encouraging' | 'cautionary' | 'celebratory';
  highlights: string[];
  statistics: KeyStatistic[];
  nextActions: SuggestedAction[];
  helpfulTips?: string[];
};

export type KeyStatistic = {
  label: string;
  value: string | number;
  trend?: 'up' | 'down' | 'stable';
  significance: 'high' | 'medium' | 'low';
};

export type SuggestedAction = {
  text: string;
  action: string;
  priority: 'high' | 'medium' | 'low';
  category: 'data' | 'analysis' | 'configuration' | 'export';
};

// Date range filtering types
export type DateRange = {
  start: Date;
  end: Date;
  preset?: 'last_7_days' | 'last_30_days' | 'last_90_days' | 'last_year' | 'custom';
};

// Dynamic insights types
export type DynamicInsight = {
  id: string;
  title: string;
  description: string;
  value: string | number;
  trend?: 'up' | 'down' | 'stable';
  significance: 'high' | 'medium' | 'low';
  category: 'trend' | 'seasonality' | 'anomaly' | 'forecast' | 'quality';
  chartData?: any[];
  actionable: boolean;
  recommendation?: string;
};

// AI-driven session insights types
export type ChatSessionData = {
  messages: ChatMessage[];
  businessUnits: BusinessUnit[];
  dataUploads: DataUploadEvent[];
  userActions: UserAction[];
  sessionStartTime: Date;
  currentState: AppState;
};

export type SessionInsights = {
  summary: string;
  keyActivities: ActivitySummary[];
  dataStatus: DataStatusSummary;
  nextSteps: RecommendedAction[];
  progressMetrics: ProgressMetric[];
};

export type ActivitySummary = {
  type: 'bu_creation' | 'lob_creation' | 'data_upload' | 'analysis_request';
  timestamp: Date;
  details: string;
  impact: 'high' | 'medium' | 'low';
};

export type DataStatusSummary = {
  totalBUs: number;
  totalLOBs: number;
  dataUploaded: boolean;
  recordCount: number;
  dataQuality?: number;
  readyForForecasting: boolean;
};

export type RecommendedAction = {
  text: string;
  action: string;
  priority: 'high' | 'medium' | 'low';
  category: 'data' | 'analysis' | 'configuration' | 'export';
};

export type ProgressMetric = {
  label: string;
  value: number;
  maxValue: number;
  unit: string;
};

export type DataUploadEvent = {
  id: string;
  fileName: string;
  recordCount: number;
  quality: number;
  timestamp: Date;
  lobId: string;
  columns: string[];
};

export type UserAction = {
  id: string;
  type: 'bu_created' | 'lob_created' | 'data_uploaded' | 'filter_applied' | 'analysis_started';
  timestamp: Date;
  data: any;
};

export type SessionEvent = {
  type: 'bu_created' | 'lob_created' | 'data_uploaded' | 'analysis_started';
  timestamp: Date;
  data: any;
};

export type AppState = {
  businessUnits: BusinessUnit[];
  selectedBuId: string | null;
  selectedLobId: string | null;
  messages: ChatMessage[];
  agents: Agent[];
  workflow: WorkflowStep[];
  dateRange?: DateRange;
  insights: DynamicInsight[];
  isInsightsPanelOpen: boolean;
  hasAnalyzedData: boolean;
  analyzedData: {
    hasEDA: boolean;
    hasForecasting: boolean;
    hasInsights: boolean;
    hasPreprocessing: boolean;
    lastAnalysisDate: Date | null;
    lastAnalysisType: 'eda' | 'forecasting' | 'comparative' | 'whatif' | null;
    outliers: OutlierData[];
    forecastData: ForecastData[];
  };
  queuedUserPrompt: string | null;
  isProcessing: boolean;
};

export type InsightsContext = {
  businessUnits: BusinessUnit[];
  hasDataUploads: boolean;
  dataUploads: DataUploadEvent[];
  hasAnalysis: boolean;
  sessionData: ChatSessionData;
};

// Value column processing types
export type ProcessedValueData = {
  values: number[];
  unit: string;
  isUnitsColumn: boolean;
  statistics: {
    min: number;
    max: number;
    average: number;
    total: number;
  };
};
